import { readFileSync, mkdirSync, writeFileSync } from "node:fs";
import { resolve, dirname, sep } from "node:path";
import type { Command } from "commander";
import { tokenize } from "../../parser/tokenizer.js";
import { parse } from "../../parser/parser.js";
import { buildGraph } from "../../parser/build-graph.js";
import { inferShapes } from "../../shape/infer.js";
import { getTarget } from "../../codegen/index.js";
import { loadPlugins } from "../../plugins/index.js";
import { registry } from "../../blocks/registry.js";
import { loadConfig } from "../../config/loader.js";

export function registerGenerateCommand(program: Command): void {
  program
    .command("generate <file>")
    .description("Generate code from a .mlmd model")
    .option("-o, --output <dir>", "output directory")
    .option("-t, --target <target>", "target framework (pytorch, keras, candle)")
    .action(async (file: string, options: { output?: string; target?: string }) => {
      const config = loadConfig();

      // Load plugins from config if present
      if (config?.plugins) {
        await loadPlugins(config.plugins);
      }

      // Read and parse the model file
      let source: string;
      try {
        source = readFileSync(resolve(file), "utf-8");
      } catch (err) {
        console.error(`generate: cannot read file: ${file}`);
        process.exit(1);
      }

      const tokens = tokenize(source);
      const parseResult = parse(tokens);
      if (parseResult.errors.length > 0) {
        for (const e of parseResult.errors) {
          console.error(`parse error: ${e.message}`);
        }
        process.exit(1);
      }

      const graph = buildGraph(parseResult.nodes);
      inferShapes(graph, registry);

      // Determine targets
      let targets: { lang: string; out: string }[];
      if (options.target) {
        targets = [{ lang: options.target, out: options.output ?? "." }];
      } else if (config?.targets && config.targets.length > 0) {
        targets = config.targets.map((t) => ({
          lang: t.lang,
          out: options.output ?? t.out,
        }));
      } else {
        // Default
        targets = [{ lang: "pytorch", out: options.output ?? "." }];
      }

      for (const { lang, out } of targets) {
        const codegenTarget = getTarget(lang);
        if (!codegenTarget) {
          console.error(`generate: unknown target "${lang}"`);
          process.exit(1);
        }

        const files = codegenTarget.generate(graph, registry);
        const outDir = resolve(out);
        mkdirSync(outDir, { recursive: true });

        for (const generated of files) {
          const filePath = resolve(outDir, generated.path);
          if (!filePath.startsWith(outDir + sep) && filePath !== outDir) {
            console.error(`generate: refusing to write outside output directory: ${generated.path}`);
            process.exit(1);
          }
          mkdirSync(dirname(filePath), { recursive: true });
          writeFileSync(filePath, generated.content, "utf-8");
          console.log(`generated: ${filePath}`);
        }
      }
    });
}
