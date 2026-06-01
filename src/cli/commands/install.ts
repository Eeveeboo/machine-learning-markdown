import { mkdirSync, cpSync, readFileSync, writeFileSync, existsSync } from "node:fs";
import { resolve, dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { homedir } from "node:os";
import { execFileSync } from "node:child_process";
import type { Command } from "commander";

function findProjectRoot(metaUrl: string): string {
  let dir = dirname(fileURLToPath(metaUrl));
  while (dir !== "/") {
    if (existsSync(join(dir, "package.json"))) {
      return dir;
    }
    dir = dirname(dir);
  }
  throw new Error("could not find package root");
}

const PROJECT_ROOT = findProjectRoot(import.meta.url);

function vscodeExtensionsDir(): string {
  return resolve(homedir(), ".vscode/extensions");
}

export function registerInstallCommand(program: Command): void {
  const install = program
    .command("install")
    .description("Install MLMD editor extensions");

  install
    .command("zed")
    .description("Install MLMD as a Zed dev extension")
    .action(() => installZed());

  install
    .command("vscode")
    .description("Install VS Code extension (~/.vscode/extensions/mlmd-vscode/)")
    .action(() => installVscode());
}

function installZed(): void {
  const extDir = resolve(PROJECT_ROOT, "zed-mlmd");
  const grammarDir = resolve(extDir, "grammars/mlmd");

  if (!existsSync(extDir)) {
    console.error("error: zed-mlmd/ not found at " + extDir);
    process.exit(1);
  }

  // 1. Install grammar dependencies
  console.log("installing grammar dependencies...");
  execFileSync("npm", ["install"], { cwd: grammarDir, stdio: "inherit" });

  // 2. Generate parser from grammar.js
  const treeSitterCmd = existsSync(join(grammarDir, "node_modules/.bin/tree-sitter"))
    ? join(grammarDir, "node_modules/.bin/tree-sitter")
    : "tree-sitter";

  console.log("generating parser...");
  execFileSync(treeSitterCmd, ["generate"], { cwd: grammarDir, stdio: "inherit" });

  console.log("\n\x1b[32m✓\x1b[0m Extension prepared at: \x1b[1m" + extDir + "\x1b[0m\n");
  console.log("To install in Zed:");
  console.log("  1. Open Zed");
  console.log("  2. Press \x1b[1mCmd+Shift+X\x1b[0m (or \x1b[1mCtrl+Shift+X\x1b[0m on Linux)");
  console.log("  3. Click \x1b[1mInstall Dev Extension\x1b[0m (top-right)");
  console.log('  4. Select the directory: \x1b[1m' + extDir + '\x1b[0m');
  console.log("  5. Open any \x1b[1m.mlmd\x1b[0m file — highlighting + LSP will activate\n");
}

function installVscode(): void {
  const extSrc = resolve(PROJECT_ROOT, "mlmd-vscode");
  const syntaxSrc = resolve(PROJECT_ROOT, "syntaxes");
  const langConfigSrc = resolve(PROJECT_ROOT, "languages/mlmd/language-configuration.json");
  const destDir = resolve(vscodeExtensionsDir(), "mlmd-vscode");

  if (!existsSync(extSrc)) {
    console.error("error: mlmd-vscode/ not found");
    process.exit(1);
  }
  if (!existsSync(syntaxSrc)) {
    console.error("error: syntaxes/ not found");
    process.exit(1);
  }

  // 1. Copy extension directory
  mkdirSync(vscodeExtensionsDir(), { recursive: true });
  cpSync(extSrc, destDir, { recursive: true, force: true });

  // 2. Copy syntaxes into extension
  const syntaxDest = resolve(destDir, "syntaxes");
  cpSync(syntaxSrc, syntaxDest, { recursive: true, force: true });

  // 3. Copy language-configuration.json from languages/mlmd/
  if (existsSync(langConfigSrc)) {
    cpSync(langConfigSrc, resolve(destDir, "language-configuration.json"), { force: true });
  }

  // 4. Fix grammar path in package.json (../syntaxes/ → ./syntaxes/)
  const pkgPath = resolve(destDir, "package.json");
  const pkg = JSON.parse(readFileSync(pkgPath, "utf-8"));
  if (pkg.contributes?.grammars) {
    for (const g of pkg.contributes.grammars) {
      if (g.path && g.path.startsWith("../")) {
        g.path = g.path.replace(/^\.\.\//, "./");
      }
    }
  }
  writeFileSync(pkgPath, JSON.stringify(pkg, null, 2) + "\n", "utf-8");

  // 5. Install deps and build
  console.log("installing dependencies...");
  execFileSync("npm", ["install"], { cwd: destDir, stdio: "inherit" });

  console.log("building extension...");
  execFileSync("npm", ["run", "build"], { cwd: destDir, stdio: "inherit" });

  console.log(`installed vs code extension to ${destDir}`);
  console.log("reload VS Code to activate (Developer: Reload Window)");
}
