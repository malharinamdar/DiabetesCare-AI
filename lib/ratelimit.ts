// Best-effort per-IP limiter for the LLM routes. In-memory, so it resets per serverless instance.

const hits = new Map<string, number[]>();

export function limited(req: Request, max = 20, windowMs = 10 * 60_000) {
  const ip = req.headers.get("x-forwarded-for")?.split(",")[0].trim() || "local";
  const now = Date.now();
  const recent = (hits.get(ip) ?? []).filter((t) => now - t < windowMs);
  recent.push(now);
  hits.set(ip, recent);
  if (hits.size > 5000) hits.clear();
  return recent.length > max;
}

export const tooMany = () =>
  Response.json({ error: "Too many requests. Give it a few minutes." }, { status: 429 });
