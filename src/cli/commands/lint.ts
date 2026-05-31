import { readFileSync } from "fs";
import type { Command } from "commander";
import { tokenize } from "../../parser/index.js";
import { parse } from "../../parser/index.js";
import { buildGraph } from "../../parser/index.js";
import { registry } from "../../blocks/registry.js";
import { lint, type LintDiagnostic } from "../../lint/index.js";
// Import side-effect: registers all built-in blocks
import "../../blocks/index.js";

function formatDiagnostic(d: LintDiagnostic): string {
  const { severity, message, loc, rule } = d;
  const locStr = `${loc.line}:${loc.col}`;
  return `${severity.toUpperCase()} [${rule}] at ${locStr}: ${message}`;
}

export function registerLintCommand(program: Command): void {
  program
    .command("lint <file>")
    .description("Lint a .mlmd file for errors and warnings")
    .option("--json", "output diagnostics as JSON")
    .action((file: string, options: { json?: boolean }) => {
      let source: string;
      try {
        source = readFileSync(file, "utf-8");
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        console.error(`Error reading file: ${msg}`);
        process.exit(1);
      }

      const tokens = tokenize(source);
      const parseResult = parse(tokens);

      if (parseResult.errors.length > 0) {
        if (options.json) {
          const jsonDiags = parseResult.errors.map((e) => ({
            severity: "error",
            message: e.message,
            loc: e.loc,
            rule: "parse-error",
          }));
          console.log(JSON.stringify(jsonDiags, null, 2));
        } else {
          for (const e of parseResult.errors) {
            const locStr = `${e.loc.line}:${e.loc.col}`;
            console.log(`ERROR [parse-error] at ${locStr}: ${e.message}`);
          }
        }
        process.exit(1);
      }

      const graph = buildGraph(parseResult.nodes);
      const diags = lint(graph, registry);

      if (options.json) {
        console.log(JSON.stringify(diags, null, 2));
      } else {
        if (diags.length === 0) {
          console.log("No issues found.");
        } else {
          for (const d of diags) {
            console.log(formatDiagnostic(d));
          }
        }
      }

      const hasErrors = diags.some((d) => d.severity === "error");
      process.exit(hasErrors ? 1 : 0);
    });
}
