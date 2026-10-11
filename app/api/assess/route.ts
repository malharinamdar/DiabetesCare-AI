import { NextResponse } from "next/server";
import { assess, parseInput } from "@/lib/model";

export async function POST(req: Request) {
  const body = await req.json().catch(() => null);
  const parsed = parseInput(body);
  if ("error" in parsed) return NextResponse.json(parsed, { status: 422 });
  return NextResponse.json(assess(parsed.value));
}
