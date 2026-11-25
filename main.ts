// Boothlord Pump.fun Paper Sprint — v1: "Early-momentum graduation scout"
//
// A paper-trading bot for the agent-fun/replay/v2 harness. It speaks the
// harness protocol through the vendored shim: the harness sends a hello line,
// then one batch line per slot (events, marks, fills, account), and this bot
// answers each batch with its orders for that slot. Orders decided on the
// batch for slot s fill at slot s + latency_slots.
//
// STRATEGY (designed for a 6-hour final window, winner-takes-all on net PnL):
//   1. SCOUT young bonding curves. Every mint is watched from the first slot
//      it is seen. A coin only becomes a candidate while it is young (first
//      ~3,000 slots, roughly the first ten minutes of its life) and only
//      after it proves real buying pressure: price up at least 8% from its
//      first-seen price, at least 8 trades observed, and buys at least 60%
//      of the volume seen so far. Buying at creation with no confirmation
//      buys every rug; waiting for deep confirmation buys the top.
//   2. SIZE small and recycle fast. Target 0.30 SOL per coin, built in
//      tranches of at most 0.10 SOL per batch (and never more than the
//      harness cap of 1% of the coin's real SOL reserves, so the order is
//      never rejected for size). At most 5 open positions at once, so at
//      most 1.5 SOL is ever deployed; the rest of the 10 SOL stays free to
//      catch the next young coin. One shot per mint: once a coin is exited
//      it is never re-entered, which keeps fee churn down.
//   3. EXIT on rules, not hope. Take profit at +50% over the blended cost
//      basis, stop loss at -20%, a trailing exit that sells when a coin
//      that peaked at +30% or better gives back 12% from its peak, and a
//      hard age exit after ~12,000 slots (~45 minutes) so dead capital is
//      freed for fresh coins. The final window is only 6 hours long;
//      holding a flat coin to the flatten is the same as losing slowly.
//   4. GUARD the downside. Equity (cash plus positions marked at the live
//      bonding-curve price) is tracked against its peak; once drawdown
//      passes 40% no new buys are opened until equity recovers, which keeps
//      the run inside the competition's 50% max-drawdown guardrail with
//      margin. A 0.5 SOL cash buffer is always kept for exit fees.
//   5. MEET the minimum bar by construction. Every buy is at least 0.01
//      SOL, and the bot trades broadly across the window's new coins, so
//      the guardrail of round trips on at least 20 distinct mints is met
//      by the normal flow of the strategy, not by a special case.
//
// DETERMINISM: every decision comes from the harness state only. No clock,
// no environment, no Math.random; coins are considered in first-seen order
// (Map insertion order), so the two grading executions produce identical
// action logs. All price comparisons use bigint cross-multiplication on
// the virtual reserves, never floats.
//
// FEES (why the thresholds sit where they do): a round trip pays about
// 1.25% of value in curve fees each way plus ~0.000105 SOL of transaction
// fees per order, so a trade needs roughly a 3% move just to break even.
// The +50% / -20% exits are set far outside that band on purpose: the
// strategy only pays the fee toll when the curve has genuinely moved.

import { runBot, LAMPORTS_PER_SOL } from "./vendor/makefun-bot.ts";
import type { Batch, BotState, OrderInput } from "./vendor/makefun-bot.ts";

// ------------------------------------------------------------------ tuning
// All thresholds are structural constants of this strategy, chosen from the
// bonding-curve mechanics above — not fitted to any one window's coins.
// (Slot rate on the dev window: ~280 slots/minute, ~16,800 slots/hour.)

const TARGET_POSITION_LAMPORTS = 300_000_000n; // 0.30 SOL per coin
const MAX_TRANCHE_LAMPORTS = 100_000_000n; // 0.10 SOL per batch per coin
const MIN_BUY_LAMPORTS = 10_000_000n; // 0.01 SOL: the guardrail round-trip floor
const MIN_REAL_RESERVES_LAMPORTS = 1_000_000_000n; // 1 SOL: a 0.01 SOL buy fits the 1% cap
const CASH_BUFFER_LAMPORTS = 500_000_000n; // 0.50 SOL never deployed, kept for exit fees
const MAX_OPEN_POSITIONS = 5; // the harness cap; never hold more
const MAX_ORDERS_PER_BATCH = 12; // under the harness cap of 16, with headroom

const ENTRY_MAX_AGE_SLOTS = 3_000; // ~11 min: only young coins qualify
const ENTRY_MIN_TRADES = 8; // enough tape to tell pressure from noise
const ENTRY_MIN_RISE_BPS = 800; // +8% over the first-seen price
const ENTRY_MIN_BUY_SHARE_BPS = 6_000; // buys >= 60% of observed volume
const ENTRY_MAX_SLIPPAGE_BPS = 500;

const TAKE_PROFIT_BPS = 15_000; // +50% over blended cost basis
const STOP_LOSS_BPS = 8_000; // -20% under blended cost basis
const TRAIL_ARM_BPS = 13_000; // trailing exit arms once the peak hits +30%
const TRAIL_GIVEBACK_BPS = 8_800; // sell when price falls to 88% of its peak
const EXIT_MAX_AGE_SLOTS = 12_000; // ~45 min: hard age exit, frees the slot
const EXIT_MAX_SLIPPAGE_BPS = 500;

const DRAWDOWN_HALT_BPS = 4_000; // halt new buys past 40% drawdown from peak equity
const DRAWDOWN_RESUME_BPS = 3_000; // resume once drawdown is back under 30%

// ------------------------------------------------------------------ helpers
// A bonding-curve price is virtual_quote_reserves / virtual_token_reserves
// (SOL per token). Ratios are compared by cross-multiplication so no float
// ever enters a decision.

function priceUpAtLeast(
  curQ: bigint,
  curT: bigint,
  baseQ: bigint,
  baseT: bigint,
  bps: number,
): boolean {
  // curQ/curT >= baseQ/baseT * (10000 + bps) / 10000
  if (curT <= 0n || baseT <= 0n || curQ <= 0n || baseQ <= 0n) return false;
  const left = curQ * baseT * 10_000n;
  const right = baseQ * curT * BigInt(10_000 + bps);
  return left >= right;
}

function priceDownToAtMost(
  curQ: bigint,
  curT: bigint,
  baseQ: bigint,
  baseT: bigint,
  bps: number,
): boolean {
  // curQ/curT <= baseQ/baseT * bps / 10000
  if (curT <= 0n || baseT <= 0n || curQ <= 0n || baseQ <= 0n) return false;
  const left = curQ * baseT * 10_000n;
  const right = baseQ * curT * BigInt(bps);
  return left <= right;
}

// ------------------------------------------------------------------ per-mint tape
interface CoinTape {
  mint: string;
  firstSlot: number;
  firstQ: bigint; // virtual quote reserves at first sight
  firstT: bigint; // virtual token reserves at first sight
  trades: number;
  buyVolume: bigint; // lamports of buy-side sol_amount seen
  sellVolume: bigint; // lamports of sell-side sol_amount seen
  enteredSlot: number | null; // slot of the first buy decision, if entered
  exited: boolean; // fully exited: never re-enter
  peakQ: bigint; // virtual quote reserves at the best price since entry
  peakT: bigint; // virtual token reserves at the best price since entry
  peakSet: boolean;
}

// ------------------------------------------------------------------ the bot
const tapes = new Map<string, CoinTape>();
let peakEquityLamports = 0n;
let buysHalted = false;

function curvePrice(curve: { virtual_quote_reserves: bigint; virtual_token_reserves: bigint }): [bigint, bigint] {
  return [curve.virtual_quote_reserves, curve.virtual_token_reserves];
}

function markValueLamports(
  state: BotState,
  mint: string,
  tokens: bigint,
): bigint {
  const curve = state.curves.get(mint);
  if (curve === undefined) return 0n;
  const [q, t] = curvePrice(curve);
  if (t <= 0n) return 0n;
  return (tokens * q) / t;
}

function equityLamports(state: BotState): bigint {
  let total = state.cash;
  for (const [mint, holding] of state.positions) {
    total += markValueLamports(state, mint, holding.token_amount);
  }
  return total;
}

function observeBatch(state: BotState, batch: Batch): void {
  for (const event of batch.events) {
    if (event.event === "CreateEvent") {
      const existing = tapes.get(event.mint);
      if (existing !== undefined) continue;
      tapes.set(event.mint, {
        mint: event.mint,
        firstSlot: event.slot,
        firstQ: BigInt(event.virtual_quote_reserves),
        firstT: BigInt(event.virtual_token_reserves),
        trades: 0,
        buyVolume: 0n,
        sellVolume: 0n,
        enteredSlot: null,
        exited: false,
        peakQ: 0n,
        peakT: 1n,
        peakSet: false,
      });
      continue;
    }
    if (event.event === "TradeEvent") {
      let tape = tapes.get(event.mint);
      if (tape === undefined) {
        tape = {
          mint: event.mint,
          firstSlot: event.slot,
          firstQ: BigInt(event.virtual_quote_reserves),
          firstT: BigInt(event.virtual_token_reserves),
          trades: 0,
          buyVolume: 0n,
          sellVolume: 0n,
          enteredSlot: null,
          exited: false,
          peakQ: 0n,
          peakT: 1n,
          peakSet: false,
        };
        tapes.set(event.mint, tape);
      }
      tape.trades += 1;
      const amount = BigInt(event.sol_amount);
      if (event.is_buy) tape.buyVolume += amount;
      else tape.sellVolume += amount;
      continue;
    }
    // CompleteEvent: the curve is over; the harness closes any position.
    // The tape stays, marked exited, so the coin is never re-entered.
    if (event.event === "CompleteEvent") {
      const tape = tapes.get(event.mint);
      if (tape !== undefined) tape.exited = true;
    }
  }
}

function pendingBuyLamports(state: BotState, mint: string): bigint {
  let total = 0n;
  for (const pending of state.pending.values()) {
    if (pending.order.side === "buy" && pending.order.mint === mint) {
      total += BigInt(pending.order.quote_lamports);
    }
  }
  return total;
}

function decideExits(state: BotState, orders: OrderInput[]): void {
  for (const [mint, holding] of state.positions) {
    const curve = state.curves.get(mint);
    if (curve === undefined || curve.complete) continue;
    const tape = tapes.get(mint);
    if (tape === undefined || tape.enteredSlot === null) continue;
    const [curQ, curT] = curvePrice(curve);
    if (curT <= 0n || curQ <= 0n) continue;
    const cost = holding.cost_lamports;
    const tokens = holding.token_amount;
    if (cost <= 0n || tokens <= 0n) continue;

    // Track the best price since entry for the trailing exit. The cost
    // basis (cost per token, fees included) is the reference: comparing
    // the live curve price against it is conservative by the entry fees.
    if (!tape.peakSet) {
      tape.peakQ = curQ;
      tape.peakT = curT;
      tape.peakSet = true;
    } else if (curQ * tape.peakT > tape.peakQ * curT) {
      tape.peakQ = curQ;
      tape.peakT = curT;
    }

    const age = state.slot - tape.enteredSlot;
    // Take profit: live price at least +50% over the cost basis.
    const hitProfit = curQ * tokens * 10_000n >= cost * curT * BigInt(TAKE_PROFIT_BPS);
    // Stop loss: live price at most 80% of the cost basis.
    const hitStop = curQ * tokens * 10_000n <= cost * curT * BigInt(STOP_LOSS_BPS);
    // Trailing: peak reached +30% over cost, price now back to 88% of peak.
    const peakArmed =
      tape.peakQ * tokens * 10_000n >= cost * tape.peakT * BigInt(TRAIL_ARM_BPS);
    const hitTrail =
      peakArmed && curQ * tape.peakT * 10_000n <= tape.peakQ * curT * BigInt(TRAIL_GIVEBACK_BPS);
    const hitAge = age >= EXIT_MAX_AGE_SLOTS;

    if (hitProfit || hitStop || hitTrail || hitAge) {
      orders.push({
        side: "sell",
        mint,
        token_amount: tokens,
        max_slippage_bps: EXIT_MAX_SLIPPAGE_BPS,
      });
      tape.exited = true;
    }
  }
}

function decideEntries(state: BotState, orders: OrderInput[]): void {
  if (buysHalted) return;
  if (state.committedPositionCount() >= MAX_OPEN_POSITIONS) return;

  for (const [mint, tape] of tapes) {
    if (orders.length >= MAX_ORDERS_PER_BATCH) break;
    if (state.committedPositionCount() >= MAX_OPEN_POSITIONS) break;
    if (tape.exited || tape.enteredSlot !== null) continue;
    if (state.positions.has(mint)) continue;

    const curve = state.curves.get(mint);
    if (curve === undefined || curve.complete) continue;
    const age = state.slot - tape.firstSlot;
    if (age < 0 || age > ENTRY_MAX_AGE_SLOTS) continue;
    if (tape.trades < ENTRY_MIN_TRADES) continue;

    const [curQ, curT] = curvePrice(curve);
    if (curT <= 0n || curQ <= 0n) continue;
    // Momentum: price up at least 8% from the first-seen price.
    if (!priceUpAtLeast(curQ, curT, tape.firstQ, tape.firstT, ENTRY_MIN_RISE_BPS)) continue;
    // Real buying pressure: buys at least 60% of observed volume.
    const volume = tape.buyVolume + tape.sellVolume;
    if (volume <= 0n) continue;
    if (tape.buyVolume * 10_000n < volume * BigInt(ENTRY_MIN_BUY_SHARE_BPS)) continue;
    // The harness caps one batch's buys on a mint at 1% of real reserves;
    // below 1 SOL of real reserves a 0.01 SOL buy cannot fit the cap.
    if (curve.real_quote_reserves < MIN_REAL_RESERVES_LAMPORTS) continue;

    // Size this tranche: at most 0.10 SOL, at most the harness cap, and
    // only up to what is left of the 0.30 SOL target (pending buys count).
    const committed = pendingBuyLamports(state, mint);
    const remaining = TARGET_POSITION_LAMPORTS - committed;
    if (remaining < MIN_BUY_LAMPORTS) continue;
    let tranche = remaining < MAX_TRANCHE_LAMPORTS ? remaining : MAX_TRANCHE_LAMPORTS;
    const cap = state.maxBuyLamports(mint);
    if (cap < tranche) tranche = cap;
    if (tranche < MIN_BUY_LAMPORTS) continue;

    // Cash check: keep the 0.5 SOL buffer plus this order's tx fee free.
    const need = tranche + state.txFeeLamports();
    if (state.freeCashLamports() - need < CASH_BUFFER_LAMPORTS) continue;

    orders.push({
      side: "buy",
      mint,
      quote_lamports: tranche,
      max_slippage_bps: ENTRY_MAX_SLIPPAGE_BPS,
    });
    tape.enteredSlot = state.slot;
  }
}

await runBot({
  onHello(state) {
    // Capital is 10 SOL; the peak starts there, so the drawdown guard has
    // a reference from the first batch onward.
    peakEquityLamports = BigInt(state.hello.capital.lamports);
    console.error(
      `[hello] capital=${state.hello.capital.lamports} seed=${state.hello.seed} ` +
        `maxPositions=${state.hello.caps.max_open_positions} ` +
        `maxOrdersPerBatch=${state.hello.caps.max_orders_per_batch}`,
    );
  },

  onBatch(state, batch) {
    observeBatch(state, batch);

    // Drawdown guard on marked equity, updated every batch.
    const equity = equityLamports(state);
    if (equity > peakEquityLamports) peakEquityLamports = equity;
    if (peakEquityLamports > 0n) {
      const drawdownBps = Number(((peakEquityLamports - equity) * 10_000n) / peakEquityLamports);
      if (!buysHalted && drawdownBps >= DRAWDOWN_HALT_BPS) {
        buysHalted = true;
        console.error(`[batch ${batch.seq}] drawdown guard: halt buys at ${drawdownBps} bps`);
      } else if (buysHalted && drawdownBps <= DRAWDOWN_RESUME_BPS) {
        buysHalted = false;
        console.error(`[batch ${batch.seq}] drawdown guard: resume buys at ${drawdownBps} bps`);
      }
    }

    const orders: OrderInput[] = [];
    decideExits(state, orders);
    decideEntries(state, orders);
    return orders;
  },

  onDone(state, done) {
    const equity = equityLamports(state);
    const pnl = equity - BigInt(state.hello.capital.lamports);
    console.error(
      `[done] reason=${done.reason} cash=${state.cash} ` +
        `positions=${state.positions.size} equity=${equity} pnlLamports=${pnl} ` +
        `coinsSeen=${tapes.size}`,
    );
  },
});
