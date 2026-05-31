import type { Command } from "commander";
import { startServer } from "../../lsp/index.js";

export function registerLspCommand(program: Command): void {
  program
    .command("lsp")
    .description("Start the Language Server Protocol server")
    .option("--stdio", "use stdio transport")
    .action((_options: Record<string, unknown>) => {
      startServer();
    });
}
