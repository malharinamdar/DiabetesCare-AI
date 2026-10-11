// ECG-style divider between the hero and the assessment.
export default function Pulse() {
  return (
    <div aria-hidden className="mb-12 flex items-center">
      <div className="h-px flex-1 bg-line" />
      <svg width="132" height="40" viewBox="0 0 132 40" className="shrink-0">
        <path
          d="M0 22 H34 L40 16 L46 22 H54 L60 4 L67 36 L73 14 L78 22 H92 L99 17 L106 22 H132"
          fill="none"
          stroke="var(--color-high)"
          strokeWidth="1.6"
          strokeLinejoin="round"
          strokeLinecap="round"
        />
      </svg>
      <div className="h-px flex-1 bg-line" />
    </div>
  );
}
