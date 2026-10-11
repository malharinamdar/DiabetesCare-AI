import type { Metadata } from "next";
import Chat from "@/components/Chat";

export const metadata: Metadata = { title: "Assistant" };

export default function AssistantPage() {
  return <Chat />;
}
