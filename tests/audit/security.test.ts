/**
 * Security boundary tests for the plugin loader.
 *
 * These tests verify that `loadPlugins` correctly enforces:
 *   1. Out-of-root rejection — directories outside the project root are refused
 *   2. Symlink skipping   — symbolic-link entries are warned and skipped
 *   3. Path-traversal prevention — filenames containing literal `..` are
 *      resolved within the plugin directory and do not escape.
 *
 * Temp directories for tests that must *pass* the CWD check are created
 * under the project root so the security check naturally succeeds.
 * Temp directories for the out-of-root test are created under `/tmp`.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { loadPlugins } from "../../src/plugins/loader.js";
import * as fs from "node:fs";
import * as path from "node:path";
import * as os from "node:os";

/** Create a unique temp directory under the project root. */
function tmpDirInProject(prefix: string): string {
  return fs.mkdtempSync(path.join(process.cwd(), prefix));
}

/** Create a unique temp directory under the OS tmpdir (outside project root). */
function tmpDirOutside(prefix: string): string {
  return fs.mkdtempSync(path.join(os.tmpdir(), prefix));
}

/** Write a minimal valid ESM plugin file that `loadPlugins` can import. */
function writePlugin(dir: string, name: string): void {
  fs.writeFileSync(
    path.join(dir, name),
    `export const ${path.parse(name).name} = {
  inputs: [{ name: "x", type: "tensor" }],
  outputs: [{ name: "y", type: "tensor" }],
};
`,
  );
}

describe("plugin loader security boundaries", () => {
  let warnSpy: any;

  beforeEach(() => {
    warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
  });

  afterEach(() => {
    warnSpy.mockRestore();
  });

  // ---------------------------------------------------------------------------
  // 1. Out-of-root rejection
  // ---------------------------------------------------------------------------
  describe("out-of-root rejection", () => {
    it("returns empty map and warns when dir is outside CWD", async () => {
      const dir = tmpDirOutside("nnml-test-outofroot-");
      try {
        const result = await loadPlugins(dir);

        expect(result.size).toBe(0);
        expect(warnSpy).toHaveBeenCalledWith(
          expect.stringContaining(
            "Refusing to load plugins from outside project root",
          ),
        );
      } finally {
        fs.rmSync(dir, { recursive: true, force: true });
      }
    });
  });

  // ---------------------------------------------------------------------------
  // 2. Symlink skipping
  // ---------------------------------------------------------------------------
  describe("symlink skipping", () => {
    it("loads real plugin files but skips symlinks with a warning", async () => {
      const base = tmpDirInProject("nnml-test-symlink-");
      try {
        writePlugin(base, "testblock.js");

        const symPath = path.join(base, "evil.mjs");
        fs.symlinkSync(path.join(base, "nonexistent.js"), symPath);

        const result = await loadPlugins(base);

        // Only the real file should be loaded
        expect(result.size).toBe(1);
        expect(result.has("testblock")).toBe(true);

        // The symlink must have been warned about
        expect(warnSpy).toHaveBeenCalledWith(
          expect.stringContaining("Skipping symlink"),
        );
      } finally {
        fs.rmSync(base, { recursive: true, force: true });
      }
    });

    it("loads non-symlink entries alongside symlinked ones", async () => {
      const base = tmpDirInProject("nnml-test-symlink2-");
      try {
        writePlugin(base, "alpha.js");
        writePlugin(base, "beta.js");

        fs.symlinkSync(path.join(base, "alpha.js"), path.join(base, "evil.mjs"));

        const result = await loadPlugins(base);

        expect(result.size).toBe(2);
        expect(result.has("alpha")).toBe(true);
        expect(result.has("beta")).toBe(true);
        expect(result.has("evil")).toBe(false);
      } finally {
        fs.rmSync(base, { recursive: true, force: true });
      }
    });
  });

  // ---------------------------------------------------------------------------
  // 3. Path-traversal prevention
  // ---------------------------------------------------------------------------
  describe("path traversal prevention", () => {
    it("handles filenames containing literal `..` — resolved within dir", async () => {
      const base = tmpDirInProject("nnml-test-traversal-");
      try {
        // A file whose *name* (not path) contains ".." as literal characters.
        // On Unix this is a legal filename — path.resolve treats it as a plain
        // component, NOT as a parent-directory reference.
        //
        // We use default export because the basename "test." is not a valid JS
        // identifier and cannot be used as a named export.
        fs.writeFileSync(
          path.join(base, "test..js"),
          `export default {
  inputs: [{ name: "x", type: "tensor" }],
  outputs: [{ name: "y", type: "tensor" }],
};
`,
        );

        const result = await loadPlugins(base);

        // The loader resolves test..js within the plugin directory; the
        // basename extracted is "test." (everything before ".js").  Since
        // there is no named export "test.", the default export is used
        // and the plugin is loaded under key "test.".
        expect(result.size).toBe(1);
        expect(result.has("test.")).toBe(true);

        // The path-traversal guard at line~65 must NOT reject this — the
        // resolved path stays inside the plugin directory (line 65 passes).
        expect(warnSpy).not.toHaveBeenCalledWith(
          expect.stringContaining("Refusing to load"),
        );
      } finally {
        fs.rmSync(base, { recursive: true, force: true });
      }
    });
  });
});
