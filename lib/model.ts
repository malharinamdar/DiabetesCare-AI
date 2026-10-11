import raw from "./model/forest.json";
import type { AssessInput, Assessment, Band, Contribution } from "./types";

type Tree = { f: number[]; t: number[]; l: number[]; r: number[]; v: number[] };
type Forest = { features: string[]; trees: Tree[]; trainPrior: number; resampledPrior: number };
const forest = raw as unknown as Forest;

// Band cut-offs on the raw forest score. 0.5 is the operating point the metrics are reported at.
const HIGH = 0.5;
const ELEVATED = 0.25;

const odds = (p: number) => p / (1 - p);
const PRIOR_SHIFT = odds(forest.trainPrior) / odds(forest.resampledPrior);

/** The forest was trained on a rebalanced (1/3 positive) sample; map its score back to real prevalence. */
export function toPopulation(score: number) {
  const s = Math.min(Math.max(score, 1e-6), 1 - 1e-6);
  const o = odds(s) * PRIOR_SHIFT;
  return o / (1 + o);
}

export function bmiOf(height: number, weight: number) {
  return weight / (height / 100) ** 2;
}

function vector(input: AssessInput, bmi: number): Record<string, number> {
  return {
    age: input.age,
    bmi,
    HbA1c_level: input.hba1c,
    blood_glucose_level: input.glucose,
    hypertension: input.hypertension ? 1 : 0,
    heart_disease: input.heartDisease ? 1 : 0,
    gender_male: input.sex === "male" ? 1 : 0,
    smoking_current: input.smoking === "current" ? 1 : 0,
    smoking_past: input.smoking === "former" ? 1 : 0,
  };
}

// One-hot columns are folded back into the question the user answered.
const GROUP: Record<string, string> = {
  age: "age",
  bmi: "bmi",
  HbA1c_level: "hba1c",
  blood_glucose_level: "glucose",
  hypertension: "hypertension",
  heart_disease: "heart",
  gender_male: "sex",
  smoking_current: "smoking",
  smoking_past: "smoking",
};

export function assess(input: AssessInput): Assessment {
  const bmi = bmiOf(input.height, input.weight);
  const named = vector(input, bmi);
  const x = forest.features.map((f) => Math.fround(named[f]));

  // Walk each tree, crediting every split with the change in P(diabetes) it causes (Saabas attribution).
  const delta: Record<string, number> = {};
  let base = 0;
  let score = 0;
  for (const tree of forest.trees) {
    let i = 0;
    base += tree.v[0];
    while (tree.l[i] !== -1) {
      const next = x[tree.f[i]] <= tree.t[i] ? tree.l[i] : tree.r[i];
      const g = GROUP[forest.features[tree.f[i]]];
      delta[g] = (delta[g] ?? 0) + tree.v[next] - tree.v[i];
      i = next;
    }
    score += tree.v[i];
  }
  const n = forest.trees.length;
  base /= n;
  score /= n;

  const shown: Record<string, [string, string]> = {
    hba1c: ["HbA1c", `${input.hba1c.toFixed(1)}%`],
    glucose: ["Blood glucose", `${Math.round(input.glucose)} mg/dL`],
    age: ["Age", `${Math.round(input.age)}`],
    bmi: ["BMI", bmi.toFixed(1)],
    hypertension: ["Hypertension", input.hypertension ? "yes" : "no"],
    heart: ["Heart disease", input.heartDisease ? "yes" : "no"],
    smoking: ["Smoking", input.smoking],
    sex: ["Sex", input.sex],
  };
  const contributions: Contribution[] = Object.entries(shown)
    .map(([key, [label, value]]) => ({ key, label, value, delta: (delta[key] ?? 0) / n }))
    .sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta));

  const band: Band = score >= HIGH ? "high" : score >= ELEVATED ? "elevated" : "low";
  return {
    bmi,
    score,
    probability: toPopulation(score),
    band,
    threshold: toPopulation(HIGH),
    base,
    contributions,
  };
}

export const LIMITS = {
  age: [1, 100],
  height: [100, 230],
  weight: [25, 250],
  hba1c: [3, 15],
  glucose: [50, 400],
} as const;

export function parseInput(body: unknown): { value: AssessInput } | { error: string } {
  if (!body || typeof body !== "object") return { error: "Expected a JSON object." };
  const b = body as Record<string, unknown>;
  if (b.sex !== "female" && b.sex !== "male") return { error: "Sex must be female or male." };
  if (b.smoking !== "never" && b.smoking !== "former" && b.smoking !== "current")
    return { error: "Smoking must be never, former or current." };
  const nums = {} as Record<keyof typeof LIMITS, number>;
  for (const [k, [lo, hi]] of Object.entries(LIMITS) as [keyof typeof LIMITS, readonly [number, number]][]) {
    const v = Number(b[k]);
    if (!Number.isFinite(v) || v < lo || v > hi) return { error: `${k} must be between ${lo} and ${hi}.` };
    nums[k] = v;
  }
  const bmi = bmiOf(nums.height, nums.weight);
  if (bmi < 10 || bmi > 80) return { error: "Height and weight give an implausible BMI." };
  return {
    value: {
      sex: b.sex,
      smoking: b.smoking,
      hypertension: b.hypertension === true,
      heartDisease: b.heartDisease === true,
      ...nums,
    },
  };
}
