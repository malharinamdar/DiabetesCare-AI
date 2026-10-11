import type { Metadata } from "next";
import metrics from "@/lib/model/metrics.json";

export const metadata: Metadata = { title: "Method" };

const pct = (v: number) => `${(v * 100).toFixed(1)}%`;

export default function Method() {
  const s = metrics.served_model;
  const u = metrics.uncapped_reference;
  const rows: [string, number, number][] = [
    ["Accuracy", s.accuracy, u.accuracy],
    ["ROC AUC", s.roc_auc, u.roc_auc],
    ["Recall (diabetic)", s.recall, u.recall],
    ["Precision (diabetic)", s.precision, u.precision],
    ["F1 (diabetic)", s.f1, u.f1],
  ];

  return (
    <article className="pt-8 sm:pt-14">
      <p className="num text-[11px] uppercase tracking-[0.16em] text-muted">Method</p>
      <h1 className="mt-5 max-w-3xl font-display text-[40px] leading-[1.05] tracking-[-0.02em] sm:text-[56px]">
        How the estimate is made
      </h1>

      <div className="mt-14 grid gap-14 lg:grid-cols-2">
        <section>
          <h2 className="font-display text-2xl">Data</h2>
          <p className="mt-3 max-w-prose leading-relaxed text-muted">
            The Kaggle diabetes prediction dataset: {metrics.rows_after_dedupe.toLocaleString("en-IN")} records after
            removing exact duplicates, {pct(metrics.prevalence)} of them diabetic. Inputs are sex, age, BMI, HbA1c,
            blood glucose, hypertension, heart disease and smoking history. Smoking is folded into never / former /
            current, as in v1.
          </p>
        </section>

        <section>
          <h2 className="font-display text-2xl">Model</h2>
          <p className="mt-3 max-w-prose leading-relaxed text-muted">
            A 100-tree random forest. Diabetic cases are oversampled with SMOTE to a 1:10 ratio, then the majority class is
            under-sampled to 1:2 before training. Trees are capped at depth 10 so the forest can be served
            from a serverless function; that costs nothing on the held-out set.
          </p>
        </section>

        <section className="lg:col-span-2">
          <h2 className="font-display text-2xl">Held-out results</h2>
          <p className="mt-3 max-w-prose leading-relaxed text-muted">
            Stratified 80/20 split, {metrics.test_size.toLocaleString("en-IN")} test records, scored at the
            screen-positive threshold. Accuracy alone flatters a dataset that is {pct(1 - metrics.prevalence)}{" "}
            non-diabetic, so recall and precision are shown too. v1 reported {pct(metrics.v1_reported_accuracy)}{" "}
            accuracy.
          </p>
          <div className="mt-6 overflow-x-auto">
            <table className="w-full min-w-[480px] text-[15px]">
              <thead>
                <tr className="num border-b border-line text-left text-[11px] uppercase tracking-[0.12em] text-faint">
                  <th className="py-3 font-normal">Metric</th>
                  <th className="py-3 text-right font-normal">Served (depth 10)</th>
                  <th className="py-3 text-right font-normal">Uncapped trees</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-line">
                {rows.map(([label, a, b]) => (
                  <tr key={label}>
                    <td className="py-2.5 text-muted">{label}</td>
                    <td className="num py-2.5 text-right">{pct(a)}</td>
                    <td className="num py-2.5 text-right text-muted">{pct(b)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        <section>
          <h2 className="font-display text-2xl">From score to probability</h2>
          <p className="mt-3 max-w-prose leading-relaxed text-muted">
            Because the forest is trained on a rebalanced sample (one in three positive), its raw score overstates
            risk. The displayed probability applies a prior-shift correction back to the real {pct(metrics.prevalence)}{" "}
            prevalence (Brier score {s.brier_population.toFixed(3)} on the test set). The screen-positive line is the
            forest&apos;s 0.5 decision point expressed on that corrected scale.
          </p>
        </section>

        <section>
          <h2 className="font-display text-2xl">Explanations</h2>
          <p className="mt-3 max-w-prose leading-relaxed text-muted">
            &ldquo;What moved the estimate&rdquo; uses path attribution (Saabas): walking each tree, every split is
            credited with the change in the node&apos;s diabetic share, and credits are summed per answer. The
            guidance and assistant use OpenAI&apos;s GPT-4.1 mini; they see your numbers but never your name.
          </p>
          <p className="mt-4 text-[14px]">
            <a
              className="underline decoration-line underline-offset-4 hover:decoration-ink"
              href="https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset"
            >
              Dataset
            </a>
            <span className="mx-2 text-faint">/</span>
            <a
              className="underline decoration-line underline-offset-4 hover:decoration-ink"
              href="https://github.com/malharinamdar/DiabetesCare-AI/blob/main/ml/train.py"
            >
              training script
            </a>
          </p>
        </section>
      </div>
    </article>
  );
}
