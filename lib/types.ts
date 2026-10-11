export type Smoking = "never" | "former" | "current";
export type Sex = "female" | "male";

export type AssessInput = {
  sex: Sex;
  age: number;
  height: number;
  weight: number;
  hba1c: number;
  glucose: number;
  hypertension: boolean;
  heartDisease: boolean;
  smoking: Smoking;
};

export type Band = "low" | "elevated" | "high";

export type Contribution = { key: string; label: string; value: string; delta: number };

export type Assessment = {
  bmi: number;
  score: number;
  probability: number;
  band: Band;
  threshold: number;
  base: number;
  contributions: Contribution[];
};

export type HistoryEntry = { id: string; at: string; input: AssessInput; result: Assessment };

export const BAND_LABEL: Record<Band, string> = { low: "Low", elevated: "Elevated", high: "High" };

export function bmiCategory(bmi: number) {
  if (bmi < 18.5) return "underweight";
  if (bmi < 25) return "healthy range";
  if (bmi < 30) return "overweight";
  return "obese range";
}
