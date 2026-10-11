import type { AssessInput, Assessment } from "./types";
import { BAND_LABEL, bmiCategory } from "./types";

export function describe(input: AssessInput, result: Assessment) {
  return [
    `Sex: ${input.sex}; age ${Math.round(input.age)}`,
    `BMI ${result.bmi.toFixed(1)} (${bmiCategory(result.bmi)}), height ${input.height} cm, weight ${input.weight} kg`,
    `HbA1c ${input.hba1c.toFixed(1)}%; blood glucose ${Math.round(input.glucose)} mg/dL`,
    `Hypertension: ${input.hypertension ? "yes" : "no"}; heart disease: ${input.heartDisease ? "yes" : "no"}; smoking: ${input.smoking}`,
    `Screening model: ${BAND_LABEL[result.band]} risk, estimated probability ${(result.probability * 100).toFixed(0)}%`,
    `Largest drivers: ${result.contributions
      .slice(0, 3)
      .map((c) => `${c.label} ${c.value} (${c.delta >= 0 ? "raises" : "lowers"} risk)`)
      .join(", ")}`,
  ].join("\n");
}

export const GUIDANCE_SYSTEM = `You are the guidance writer for DiabetesCare, a diabetes risk screening tool used mostly in India.
You receive one person's measurements and the screening model's output. Write personal, specific guidance.

Structure, using markdown headings (###):
### What your numbers say  — 2-4 sentences interpreting HbA1c, glucose and BMI against standard ADA cut-offs (HbA1c 5.7% prediabetes, 6.5% diabetes; BMI 25 / 30). Use the person's actual values.
### Food  — 4-6 concrete bullets grounded in everyday Indian meals (e.g. swaps for white rice, roti portions, dal, sabzi, snacks, chai with sugar).
### Movement  — 3-4 bullets with amounts (minutes per week, steps, strength sessions).
### Tests to ask for  — 2-4 bullets: which confirmatory tests to discuss with a doctor and how soon, scaled to the risk band.

Rules: be direct and warm; no filler; never invent phone numbers, hospital names or links; do not prescribe or dose medication; under 350 words.`;

export const CHAT_SYSTEM = `You are the DiabetesCare assistant. Answer questions about diabetes, prediabetes, blood sugar, HbA1c, diet, exercise, medication classes and related metabolic health.
Be accurate, specific and concise (usually under 180 words). Use markdown lists when they help.
Prefer guidance that fits Indian diets and context when relevant. Never invent phone numbers, hospital names or links. Do not give individual medication doses.
If something sounds urgent (very high or very low sugar with symptoms, chest pain, confusion), say plainly to seek emergency care.
If a question is unrelated to health, say briefly that you only cover diabetes and metabolic health.`;
