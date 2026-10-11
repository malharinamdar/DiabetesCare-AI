type Props = {
  title: string;
  unit: string;
  edges: number[];
  pos: number[];
  neg: number[];
  value: number;
  refs?: { at: number; label: string }[];
};

const W = 300;
const H = 96;
const TOP = 18;

export default function Distribution({ title, unit, edges, pos, neg, value, refs = [] }: Props) {
  const lo = edges[0];
  const hi = edges[edges.length - 1];
  const x = (v: number) => ((Math.min(hi, Math.max(lo, v)) - lo) / (hi - lo)) * W;
  const peak = Math.max(...pos, ...neg);
  const y = (d: number) => H - (d / peak) * (H - TOP);

  const area = (d: number[]) => {
    let p = `M0 ${H}`;
    d.forEach((v, i) => (p += ` L${x(edges[i])} ${y(v)} L${x(edges[i + 1])} ${y(v)}`));
    return `${p} L${W} ${H} Z`;
  };

  return (
    <figure>
      <figcaption className="flex items-baseline justify-between text-[13px]">
        <span>{title}</span>
        <span className="num text-xs text-muted">
          you: {value % 1 ? value.toFixed(1) : value} {unit}
        </span>
      </figcaption>
      <svg viewBox={`0 0 ${W} ${H + 16}`} className="mt-2 w-full overflow-visible" role="img" aria-label={`${title} distribution`}>
        <path d={area(neg)} fill="var(--color-wash)" />
        <path d={area(pos)} fill="var(--color-high)" fillOpacity="0.14" stroke="var(--color-high)" strokeWidth="1" strokeOpacity="0.7" />
        {refs.map((r) => (
          <g key={r.at}>
            <line x1={x(r.at)} x2={x(r.at)} y1={TOP - 6} y2={H} stroke="var(--color-faint)" strokeDasharray="2 3" />
            <text x={x(r.at) + 3} y={TOP - 1} fontSize="9" fill="var(--color-faint)" className="num">
              {r.label}
            </text>
          </g>
        ))}
        <line x1={x(value)} x2={x(value)} y1={4} y2={H} stroke="var(--color-ink)" strokeWidth="1.5" />
        <circle cx={x(value)} cy={4} r="3" fill="var(--color-ink)" />
        <line x1="0" x2={W} y1={H} y2={H} stroke="var(--color-line)" />
        <text x="0" y={H + 13} fontSize="9" fill="var(--color-faint)" className="num">
          {lo}
        </text>
        <text x={W} y={H + 13} fontSize="9" fill="var(--color-faint)" textAnchor="end" className="num">
          {hi}
        </text>
      </svg>
    </figure>
  );
}
