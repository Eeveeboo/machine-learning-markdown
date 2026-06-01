import { readFileSync, writeFileSync } from "fs";
import { resolve } from "path";
import type { Command } from "commander";
import { tokenize, parse, buildGraph } from "../../parser/index.js";
import { inferShapes } from "../../shape/infer.js";
import { loadPlugins } from "../../plugins/index.js";
import { loadConfig } from "../../config/loader.js";
import { layout } from "../../visualize/layout.js";
import { render } from "../../visualize/renderer.js";
import "../../blocks/index.js";
import { registry } from "../../blocks/registry.js";

export function registerVisualizeCommand(program: Command): void {
  program
    .command("visualize <file>")
    .description("Visualize a .mlmd model architecture as SVG")
    .option("-o, --output <file>", "output file path")
    .option("-f, --format <format>", "output format (svg)", "svg")
    .action(async (file: string, options: Record<string, string>) => {
      // Load plugins from config if present
      const config = loadConfig();
      if (config?.plugins) {
        await loadPlugins(config.plugins);
      }

      let source: string;
      try {
        source = readFileSync(file, "utf-8");
      } catch (e) {
        console.error(`Error reading file: ${file}`);
        process.exit(1);
      }

      const tokens = tokenize(source);
      const { nodes: ast } = parse(tokens);
      const graph = buildGraph(ast);
      const { graph: shapedGraph } = inferShapes(graph, registry);
      const layoutResult = layout(shapedGraph);
      const svg = render(shapedGraph, layoutResult);

      const outputPath = options["output"];
      if (outputPath) {
        writeFileSync(outputPath, svg, "utf-8");
        console.log(`SVG written to ${outputPath}`);
      } else {
        process.stdout.write(svg);
      }
    });
}
