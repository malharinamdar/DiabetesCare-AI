import { errorResponse, streamText, textResponse } from "@/lib/llm";
import { assess, parseInput } from "@/lib/model";
import { describe, GUIDANCE_SYSTEM } from "@/lib/prompt";
import { limited, tooMany } from "@/lib/ratelimit";

export const maxDuration = 60;

export async function POST(req: Request) {
  if (limited(req)) return tooMany();
  const parsed = parseInput(await req.json().catch(() => null));
  if ("error" in parsed) return Response.json(parsed, { status: 422 });

  // Recompute on the server rather than trusting a client-supplied result.
  const result = assess(parsed.value);
  try {
    const stream = await streamText({
      system: GUIDANCE_SYSTEM,
      messages: [{ role: "user", content: describe(parsed.value, result) }],
    });
    return textResponse(stream);
  } catch (err) {
    return errorResponse(err);
  }
}
