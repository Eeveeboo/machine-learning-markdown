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
  // The extension directory is the project root (where extension.toml lives)
  const extDir = PROJECT_ROOT;
  const zedMlmdDir = resolve(PROJECT_ROOT, "zed-mlmd");
  const grammarRepoDir = resolve(PROJECT_ROOT, "grammars/mlmd-grammar");
  const extensionToml = resolve(extDir, "extension.toml");

  if (!existsSync(extDir)) {
    console.error("error: project root not found at " + extDir);
    process.exit(1);
  }
  if (!existsSync(zedMlmdDir)) {
    console.error("error: zed-mlmd/ not found at " + zedMlmdDir);
    process.exit(1);
  }
  if (!existsSync(grammarRepoDir)) {
    console.error("error: grammars/mlmd-grammar/ not found at " + grammarRepoDir);
    process.exit(1);
  }

  // 1. Ensure grammar parser C source is up to date
  const treeSitterCmd = existsSync(join(grammarRepoDir, "node_modules/.bin/tree-sitter"))
    ? join(grammarRepoDir, "node_modules/.bin/tree-sitter")
    : "tree-sitter";
  if (existsSync(join(grammarRepoDir, "node_modules"))) {
    console.log("ensuring grammar dependencies...");
    execFileSync("npm", ["install"], { cwd: grammarRepoDir, stdio: "inherit" });
  }
  console.log("generating grammar parser...");
  execFileSync(treeSitterCmd, ["generate"], { cwd: grammarRepoDir, stdio: "inherit" });

  // 2. Build grammar WASM (optional — tree-sitter may not have WASM deps)
  try {
    console.log("building grammar WASM...");
    execFileSync(treeSitterCmd, ["build", "--wasm"], { cwd: grammarRepoDir, stdio: "inherit" });
    // Copy output (tree-sitter emits <name>.wasm or tree-sitter-<name>.wasm)
    const wasmCandidate = resolve(grammarRepoDir, "tree-sitter-mlmd.wasm");
    if (existsSync(wasmCandidate)) {
      cpSync(wasmCandidate, resolve(extDir, "grammars/mlmd.wasm"));
    }
  } catch {
    console.log("  (tree-sitter build --wasm unavailable; Zed will compile grammar from source)");
  }

  // 3. Update extension.toml with correct file:// URL for this machine
  console.log("configuring extension.toml...");
  let toml = readFileSync(extensionToml, "utf-8");
  const grammarUrl = "file://" + grammarRepoDir;

  if (toml.includes("[grammars.mlmd]")) {
    // Replace repository line within existing [grammars.mlmd] section
    toml = toml.replace(
      /(\[grammars\.mlmd\]\s*\n\s*repository\s*=\s*)"[^"]*"/,
      `$1"${grammarUrl}"`,
    );
    // Ensure rev is set to "main"
    if (!toml.match(/\[grammars\.mlmd\][\s\S]*?\brev\s*=\s*"/)) {
      toml = toml.replace(
        /(\[grammars\.mlmd\]\s*\n)/,
        `$1rev = "main"\n`,
      );
    }
  } else {
    // Add grammar section before [language_servers]
    const grammarSection = `[grammars.mlmd]\nrepository = "${grammarUrl}"\nrev = "main"\n`;
    if (toml.includes("[language_servers]")) {
      toml = toml.replace(/\[language_servers\]/, grammarSection + "\n[language_servers]");
    } else {
      toml += "\n" + grammarSection + "\n";
    }
  }
  writeFileSync(extensionToml, toml);

  // 4. Build extension WASM (Rust component)
  console.log("building extension WASM (cargo)...");
  execFileSync("cargo", ["build", "--target", "wasm32-wasip2", "--release"], {
    cwd: zedMlmdDir,
    stdio: "inherit",
  });

  // 5. Copy extension WASM to extension root and zed-mlmd/
  const wasmSrc = resolve(zedMlmdDir, "target/wasm32-wasip2/release/zed_mlmd.wasm");
  cpSync(wasmSrc, resolve(extDir, "extension.wasm"));
  cpSync(wasmSrc, resolve(zedMlmdDir, "extension.wasm"));

  console.log("\n\x1b[32m✓\x1b[0m Extension ready at: \x1b[1m" + extDir + "\x1b[0m\n");
  console.log("To install in Zed:");
  console.log("  1. Open Zed");
  console.log("  2. Press \x1b[1mCmd+Shift+X\x1b[0m (or \x1b[1mCtrl+Shift+X\x1b[0m on Linux)");
  console.log("  3. Click \x1b[1mInstall Dev Extension\x1b[0m (top-right)");
  console.log('  4. Select the directory: \x1b[1m' + extDir + '\x1b[0m');
  console.log("  5. Open any \x1b[1m.mlmd\x1b[0m file — highlighting + LSP will activate\n");
  console.log("Note: first load triggers grammar compilation in Zed (requires git + WASI SDK).");
  console.log("      This is a one-time build that works fully offline.\n");
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

  // 4. Install deps and build
  console.log("installing dependencies...");
  execFileSync("npm", ["install"], { cwd: destDir, stdio: "inherit" });

  console.log("building extension...");
  execFileSync("npm", ["run", "build"], { cwd: destDir, stdio: "inherit" });

  console.log(`installed vs code extension to ${destDir}`);
  console.log("reload VS Code to activate (Developer: Reload Window)");
}
