import { expect, test, type Page, type Route } from "@playwright/test";

function json(route: Route, body: unknown, status = 200) {
  return route.fulfill({
    status,
    contentType: "application/json",
    body: JSON.stringify(body),
  });
}

async function mockAPI(page: Page) {
  await page.route("**/api/**", async (route) => {
    const { pathname } = new URL(route.request().url());

    if (pathname === "/api/market/overview") {
      return json(route, {
        vix: { spot: 18.25, "1d_return": -1.2 },
        yields: { spot: 4.15, "1d_return": 0.03 },
        futures: { ES: { last: 6000, change_pct: 0.4 } },
      });
    }
    if (pathname === "/api/news") {
      return json(route, [
        {
          title: "Options positioning remains elevated",
          link: "https://example.test/news/1",
          source: "Example Wire",
          date: "2026-07-11",
        },
      ]);
    }
    if (pathname === "/api/ticker/SPY/expirations") {
      return json(route, ["2026-08-21"]);
    }
    if (pathname === "/api/flow/feed") {
      return json(route, {
        as_of: "2026-07-11",
        stale: false,
        unavailable_history: false,
        unavailable_tickers: [],
        unavailable_history_tickers: [],
        rows: [
          {
            ticker: "SPY",
            expiration_date: "2026-08-21",
            strike: 600,
            option_type: "call",
            snapshot_date: "2026-07-11",
            volume: 5000,
            open_interest: 1000,
            volume_oi: 5,
            oi_change: 250,
            mid_iv: 0.2,
            iv_change: 0.01,
            history_available: true,
            score: 250.5,
          },
        ],
      });
    }
    if (pathname === "/api/flow/hottest-chains") {
      return json(route, {
        as_of: "2026-07-11",
        stale: false,
        unavailable_history: false,
        unavailable_tickers: [],
        unavailable_history_tickers: [],
        rows: [
          {
            ticker: "SPY",
            expiration_date: "2026-08-21",
            snapshot_date: "2026-07-11",
            contracts: 120,
            total_volume: 50_000,
            total_open_interest: 200_000,
            volume_oi: 0.25,
            oi_change: 1500,
            iv_change: 0.004,
            history_available: true,
            score: 180.2,
          },
        ],
      });
    }
    if (pathname === "/api/ticker/SPY/snapshot") {
      return json(route, { symbol: "SPY", last: 600, change: 2, change_percentage: 0.33 });
    }
    if (pathname === "/api/ticker/SPY/candles") {
      return json(route, [
        { date: "2026-07-09", open: 595, high: 601, low: 594, close: 600, volume: 1000 },
        { date: "2026-07-10", open: 600, high: 603, low: 599, close: 602, volume: 1200 },
      ]);
    }
    if (pathname === "/api/ticker/SPY/term-structure") {
      return json(route, { symbol: "SPY", spot: 600, points: [{ expiration: "2026-08-21", atm_iv: 0.2 }] });
    }
    if (pathname === "/api/history/SPY/iv-rank") return json(route, { detail: "Not found" }, 404);
    if (pathname === "/api/history/metrics") return json(route, []);
    if (pathname === "/api/ticker/SPY/gex") {
      return json(route, {
        symbol: "SPY", expiration: "2026-08-21", spot: 600, offset: 35,
        profile: [{ strike: 600, net_gex: 1_000_000 }], interpretation: ["Positive gamma"],
        gamma_gap: { magnet_strike: 600, magnet_gex: 1_000_000, distance: 0, distance_pct: 0, score: 80, positive_zone: true, band_low: 595, band_high: 605 },
      });
    }
    if (pathname === "/api/ticker/SPY/exposure") {
      return json(route, { symbol: "SPY", expirations: ["2026-08-21"], spot: 600, offset: 35, risk_free_rate: 0.04, points: [{ strike: 600, vanna: 1, charm: 1 }] });
    }
    if (pathname === "/api/ticker/SPY/max-pain") {
      return json(route, { symbol: "SPY", expiration: "2026-08-21", spot: 600, max_pain: 600, curve: [{ strike: 600, call_pain: 1, put_pain: 1, total_pain: 2 }] });
    }
    if (pathname === "/api/ticker/SPY/skew") {
      return json(route, { symbol: "SPY", expiration: "2026-08-21", points: [{ strike: 600, iv_call: 0.2, iv_put: 0.21, iv_skew: 0.01 }] });
    }

    return json(route, { detail: `Unhandled mock route: ${pathname}` }, 404);
  });
}

test.beforeEach(async ({ page }) => mockAPI(page));

test("market overview renders intercepted data", async ({ page }) => {
  await page.goto("/market");
  await expect(page.getByRole("heading", { name: "Market Overview" })).toBeVisible();
  await expect(page.getByText("18.25", { exact: true })).toBeVisible();
});

test("news renders intercepted headlines", async ({ page }) => {
  await page.goto("/news");
  await expect(page.getByRole("heading", { name: "Market News" })).toBeVisible();
  await expect(page.getByText("Options positioning remains elevated")).toBeVisible();
});

test("flow renders the cached feed and hottest chains", async ({ page }) => {
  await page.goto("/flow");
  await expect(
    page.getByRole("heading", { name: "Options Positioning Proxy" }),
  ).toBeVisible();
  await expect(page.getByText("Hottest Chains")).toBeVisible();
  await expect(page.getByText("Contract-Level Feed")).toBeVisible();
  await expect(page.getByText("Current cache")).toBeVisible();
  await expect(page.getByRole("cell", { name: "600.00" })).toBeVisible();
});

test("ticker page renders its intercepted price chart and GEX", async ({ page }) => {
  await page.goto("/stock/SPY");
  await expect(page.getByRole("heading", { name: "SPY", exact: true })).toBeVisible();
  await expect(page.locator('svg[viewBox="0 0 760 300"]')).toBeVisible();
  await page.getByRole("button", { name: "GEX" }).click();
  await expect(page.getByText("Net GEX by strike — 2026-08-21")).toBeVisible();
});
