import { readFileSync } from "fs";
import { resolve } from "path";
import type { Command } from "commander";
import { tokenize } from "../../parser/index.js";
import { parse } from "../../parser/index.js";
import { buildGraph } from "../../parser/index.js";
import { inferShapes } from "../../shape/infer.js";
import { registry } from "../../blocks/registry.js";
import { lint, type LintDiagnostic } from "../../lint/index.js";
import { loadPlugins } from "../../plugins/index.js";
import { loadConfig } from "../../config/loader.js";
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
    .action(async (file: string, options: { json?: boolean }) => {
      // Load plugins from config if present
      const config = loadConfig();
      if (config?.plugins) {
        await loadPlugins(config.plugins);
      }

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
      const shapeResult = inferShapes(graph, registry);

      // Collect shape errors too
      const allDiags: LintDiagnostic[] = shapeResult.errors.map((e) => ({
        severity: "error",
        message: e.message,
        loc: { line: 0, col: 0, offset: 0 },
        rule: "shape-mismatch",
      }));

      const lintDiags = lint(graph, registry);
      allDiags.push(...lintDiags);

      if (options.json) {
        console.log(JSON.stringify(allDiags, null, 2));
      } else {
        if (allDiags.length === 0) {
          console.log("No issues found.");
        } else {
          for (const d of allDiags) {
            console.log(formatDiagnostic(d));
          }
        }
      }

      const hasErrors = allDiags.some((d) => d.severity === "error");
      process.exit(hasErrors ? 1 : 0);
    });
}
