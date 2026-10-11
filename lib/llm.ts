// Minimal OpenAI client (no SDK). Streams plain text out of the chat-completions SSE response.

const URL = "https://api.openai.com/v1/chat/completions";
const MODEL = process.env.OPENAI_MODEL || "gpt-4.1-mini";

export type Message = { role: "user" | "assistant"; content: string };

export class MissingKeyError extends Error {}

export async function streamText({
  system,
  messages,
  temperature = 0.4,
}: {
  system: string;
  messages: Message[];
  temperature?: number;
}) {
  const key = process.env.OPENAI_API_KEY;
  if (!key) throw new MissingKeyError("OPENAI_API_KEY is not configured.");

  const res = await fetch(URL, {
    method: "POST",
    headers: { "content-type": "application/json", authorization: `Bearer ${key}` },
    body: JSON.stringify({
      model: MODEL,
      stream: true,
      temperature,
      max_tokens: 1500,
      messages: [{ role: "system", content: system }, ...messages],
    }),
  });
  if (!res.ok || !res.body) {
    const detail = (await res.text().catch(() => "")).slice(0, 300);
    throw new Error(`OpenAI returned ${res.status}. ${detail}`);
  }

  const decoder = new TextDecoder();
  const encoder = new TextEncoder();
  let buffer = "";
  return res.body.pipeThrough(
    new TransformStream<Uint8Array, Uint8Array>({
      transform(chunk, controller) {
        buffer += decoder.decode(chunk, { stream: true });
        let nl: number;
        while ((nl = buffer.indexOf("\n")) >= 0) {
          const line = buffer.slice(0, nl).trim();
          buffer = buffer.slice(nl + 1);
          if (!line.startsWith("data:") || line === "data: [DONE]") continue;
          try {
            const text = JSON.parse(line.slice(5)).choices?.[0]?.delta?.content;
            if (text) controller.enqueue(encoder.encode(text));
          } catch {
            // Ignore keep-alives and partial frames.
          }
        }
      },
    }),
  );
}

export function textResponse(stream: ReadableStream<Uint8Array>) {
  return new Response(stream, {
    headers: { "content-type": "text/plain; charset=utf-8", "cache-control": "no-store" },
  });
}

export function errorResponse(err: unknown) {
  if (err instanceof MissingKeyError)
    return Response.json({ error: "The assistant is not configured on this deployment." }, { status: 503 });
  console.error(err);
  return Response.json({ error: "The language model is unavailable right now. Try again shortly." }, { status: 502 });
}
