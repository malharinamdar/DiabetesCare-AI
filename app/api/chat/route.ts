import { errorResponse, streamText, textResponse } from "@/lib/llm";
import { assess, parseInput } from "@/lib/model";
import { CHAT_SYSTEM, describe } from "@/lib/prompt";
import { limited, tooMany } from "@/lib/ratelimit";

export const maxDuration = 60;

type Msg = { role: "user" | "assistant"; content: string };

export async function POST(req: Request) {
  if (limited(req, 40)) return tooMany();
  const body = (await req.json().catch(() => null)) as { messages?: Msg[]; profile?: unknown } | null;
  const messages = (body?.messages ?? [])
    .filter((m) => (m.role === "user" || m.role === "assistant") && typeof m.content === "string")
    .slice(-16)
    .map((m) => ({ ...m, content: m.content.slice(0, 4000) }));
  if (!messages.length || messages[messages.length - 1].role !== "user")
    return Response.json({ error: "Send at least one user message." }, { status: 422 });

  let system = CHAT_SYSTEM;
  if (body?.profile) {
    const parsed = parseInput(body.profile);
    if ("value" in parsed)
      system += `\n\nThe user shared their latest screening (use it when relevant, don't repeat it back unprompted):\n${describe(parsed.value, assess(parsed.value))}`;
  }

  try {
    return textResponse(await streamText({ system, messages, temperature: 0.5 }));
  } catch (err) {
    return errorResponse(err);
  }
}
