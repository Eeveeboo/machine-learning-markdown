import { Command } from "commander";
import { registerVisualizeCommand } from "./commands/visualize.js";
import { registerGenerateCommand } from "./commands/generate.js";
import { registerLintCommand } from "./commands/lint.js";
import { registerLspCommand } from "./commands/lsp.js";
import { registerInstallCommand } from "./commands/install.js";

export function createCLI(): Command {
  const program = new Command();

  program
    .name("mlmd")
    .description("CLI for describing and validating neural network architectures")
    .version("0.1.0");

  registerVisualizeCommand(program);
  registerGenerateCommand(program);
  registerLintCommand(program);
  registerLspCommand(program);
  registerInstallCommand(program);

  return program;
}
