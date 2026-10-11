import ReactMarkdown from "react-markdown";

export default function Markdown({ text, streaming }: { text: string; streaming?: boolean }) {
  return (
    <div className="md">
      <ReactMarkdown
        components={{
          h1: "h3",
          h2: "h3",
          h4: "h3",
          // Model output never gets clickable links.
          a: ({ children }) => <span>{children}</span>,
          img: () => null,
        }}
      >
        {text}
      </ReactMarkdown>
      {streaming && <span className="caret text-faint" aria-hidden />}
    </div>
  );
}
