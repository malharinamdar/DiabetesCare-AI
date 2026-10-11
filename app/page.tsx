import AssessTool from "@/components/AssessTool";
import GlucoseDay from "@/components/GlucoseDay";
import Pulse from "@/components/Pulse";
import metrics from "@/lib/model/metrics.json";

export default function Home() {
  return (
    <>
      <section className="grid items-center gap-10 pt-8 pb-12 sm:pt-14 sm:pb-14 lg:grid-cols-[minmax(0,7fr)_minmax(0,5fr)] lg:gap-14">
        <div>
        <p className="num text-[11px] uppercase tracking-[0.16em] text-muted">
          Type 2 diabetes risk screen <span className="text-faint">·</span>{" "}
          {metrics.rows_after_dedupe.toLocaleString("en-IN")} patient records
        </p>
        <h1 className="mt-5 max-w-3xl font-display text-[44px] leading-[1.02] tracking-[-0.02em] sm:text-[62px]">
          Know where your numbers <em>stand.</em>
        </h1>
        <p className="mt-6 max-w-xl text-[17px] leading-relaxed text-muted">
          Eight routine measurements, a random forest trained on real patient records, and a clear reading of which
          answers are driving the estimate. Then personal guidance, if you want it.
        </p>
        </div>
        <div className="mx-auto w-full max-w-md lg:max-w-none">
          <GlucoseDay />
        </div>
      </section>
      <Pulse />
      <AssessTool />
    </>
  );
}
