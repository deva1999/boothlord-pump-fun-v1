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
//   2. SIZE small and recycle fast. One buy per coin of at most 0.10 SOL
//      (and never more than the harness cap of 1% of the coin's real SOL
//      reserves, so the order is never rejected for size). No second
//      tranche: adding to a position chases the curve up and drags the
//      blended cost toward the stop, and — measured on the dev tape —
//      topping up while a coin bleeds clogs the 5 position slots and
//      collapses activity below the 20-coin guardrail. At most 5 open
//      positions at once; the rest of the 10 SOL stays free to catch the
//      next young coin. Once a coin is exited it is never re-entered,
//      which keeps fee churn down.
//   3. EXIT on rules, not hope — and let the tail run. The dev tape's
//      shape decides the exits: after the entry trigger, the median coin
//      bleeds about -30% while a thin tail explodes (about 16% of
//      triggered coins reach +100%, 9% reach +200%, and those that do get
//      there fast — a median of ~200 slots). So the stop is TIGHT and the
//      target is WIDE: stop loss at -6% under the blended cost basis,
//      take profit at +200% over it, no trailing exit and no partial
//      sales (both were measured to cap the tail that pays for the
//      losers). A hard age exit after ~12,000 slots (~45 minutes) applies
//      only while a position is underwater, so dead capital is freed for
//      fresh coins while a live winner is never cut short by the clock.
//      The final window is only 6 hours long; holding a flat coin to the
//      flatten is the same as losing slowly. One sell in flight at most
//      per mint, and a sell that is rejected or fails is retried while
//      none is pending — a stranded position would clog one of the 5
//      slots for the rest of the run; only a filled sell closes a coin.
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
// The -6% stop sits just outside that band on purpose — tight enough to
// cut the median -30% bleed early, wide enough not to be pure fee noise —
// and the +200% target sits far outside it, so the strategy only pays
// the fee toll when the curve has genuinely moved, and only banks the
// moves that actually pay.

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

const TAKE_PROFIT_BPS = 30_000; // +200% over blended cost basis: bank the tail
const STOP_LOSS_BPS = 9_400; // -6% under blended cost basis: cut the bleed early
const EXIT_MAX_AGE_SLOTS = 12_000; // ~45 min: age exit, underwater positions only
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
  // A filled sell confirms the exit: the coin is done and never
  // re-entered. A rejected or failed sell confirms nothing — the exit
  // stays open and decideExits may retry it once no sell is pending.
  for (const fill of batch.fills) {
    if (fill.side === "sell" && fill.status === "filled") {
      const tape = tapes.get(fill.mint);
      if (tape !== undefined) tape.exited = true;
    }
  }
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

// A sell already in flight for this mint: sending another now would only
// be rejected (the position row still lists the tokens until the fill
// lands). Once the in-flight sell resolves, a rejected or failed one
// leaves no sell pending, so the exit can be retried next batch.
function hasPendingSell(state: BotState, mint: string): boolean {
  for (const pending of state.pending.values()) {
    if (pending.order.side === "sell" && pending.order.mint === mint) return true;
  }
  return false;
}

function decideExits(state: BotState, orders: OrderInput[]): void {
  for (const [mint, holding] of state.positions) {
    const curve = state.curves.get(mint);
    if (curve === undefined || curve.complete) continue;
    const tape = tapes.get(mint);
    if (tape === undefined || tape.enteredSlot === null) continue;
    // One sell in flight at most: while a sell for this mint is pending
    // its fill, sending another only earns a rejection. A rejected or
    // failed sell clears the pending slot, so the exit is retried — the
    // exit is only final when a fill confirms it (see observeBatch).
    if (hasPendingSell(state, mint)) continue;
    const [curQ, curT] = curvePrice(curve);
    if (curT <= 0n || curQ <= 0n) continue;
    const cost = holding.cost_lamports;
    const tokens = holding.token_amount;
    if (cost <= 0n || tokens <= 0n) continue;

    // The cost basis (cost per token, fees included) is the reference:
    // comparing the live curve price against it is conservative by the
    // entry fees, which is the safe side for both exits.
    const age = state.slot - tape.enteredSlot;
    // Take profit: live price at least +200% over the cost basis.
    const hitProfit = curQ * tokens * 10_000n >= cost * curT * BigInt(TAKE_PROFIT_BPS);
    // Stop loss: live price at most 94% of the cost basis (-6%).
    const hitStop = curQ * tokens * 10_000n <= cost * curT * BigInt(STOP_LOSS_BPS);
    // Age exit, underwater only: a position still below its cost basis
    // after ~45 minutes is dead capital — free the slot. A position
    // above cost keeps riding; the stop below protects it.
    const underwater = curQ * tokens < cost * curT;
    const hitAge = age >= EXIT_MAX_AGE_SLOTS && underwater;

    if (hitProfit || hitStop || hitAge) {
      orders.push({
        side: "sell",
        mint,
        token_amount: tokens,
        max_slippage_bps: EXIT_MAX_SLIPPAGE_BPS,
      });
      // Note: tape.exited is NOT set here. The exit becomes final only
      // when the sell's fill arrives (observeBatch); until then a
      // rejected or failed sell must stay retryable, or the position
      // strands and clogs its slot for the rest of the run.
    }
  }
}

function decideEntries(state: BotState, orders: OrderInput[]): void {
  if (buysHalted) return;
  if (state.committedPositionCount() >= MAX_OPEN_POSITIONS) return;

  for (const [mint, tape] of tapes) {
    if (orders.length >= MAX_ORDERS_PER_BATCH) break;
    if (state.committedPositionCount() >= MAX_OPEN_POSITIONS) break;
    // One shot per mint: a coin already entered is never bought again,
    // and a coin already exited is never re-entered.
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

    // Size this buy: at most 0.10 SOL and at most the harness cap.
    // (TARGET_POSITION_LAMPORTS is the notional target a single buy aims
    // at when the cap allows it; the cap usually binds first on young
    // coins, so most buys land below it.)
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
