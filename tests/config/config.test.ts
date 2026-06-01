import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { writeFileSync, mkdirSync, rmSync, mkdtempSync } from "node:fs";
import { resolve, join } from "node:path";
import { tmpdir } from "node:os";
import { loadConfig } from "../../src/config/loader.js";
import { validateConfig } from "../../src/config/schema.js";

const TMP = mkdtempSync(join(tmpdir(), "config-test-"));

describe("validateConfig", () => {
  it("accepts valid full config", () => {
    const result = validateConfig({ plugins: "./plugins", targets: [{ lang: "pytorch", out: "dist" }] });
    expect(result).toEqual({ plugins: "./plugins", targets: [{ lang: "pytorch", out: "dist" }] });
  });

  it("accepts empty config", () => {
    expect(validateConfig({})).toEqual({});
  });

  it("rejects non-object", () => {
    expect(validateConfig("bad")).toBeNull();
    expect(validateConfig(null)).toBeNull();
    expect(validateConfig([1, 2])).toBeNull();
  });

  it("rejects plugins non-string", () => {
    expect(validateConfig({ plugins: 42 })).toBeNull();
  });

  it("rejects targets non-array", () => {
    expect(validateConfig({ targets: "bad" })).toBeNull();
  });

  it("rejects target missing lang", () => {
    expect(validateConfig({ targets: [{ out: "dist" }] })).toBeNull();
  });

  it("rejects target missing out", () => {
    expect(validateConfig({ targets: [{ lang: "pytorch" }] })).toBeNull();
  });
});

describe("loadConfig", () => {
  beforeEach(() => {
    mkdirSync(TMP, { recursive: true });
    delete process.env["MLMD_CONFIG"];
  });

  afterEach(() => {
    rmSync(TMP, { recursive: true, force: true });
    delete process.env["MLMD_CONFIG"];
  });

  it("returns null when no .mlmdrc exists", () => {
    const result = loadConfig(TMP);
    expect(result).toBeNull();
  });

  it("loads .mlmdrc from cwd", () => {
    writeFileSync(resolve(TMP, ".mlmdrc"), JSON.stringify({ targets: [{ lang: "keras", out: "out" }] }));
    const result = loadConfig(TMP);
    expect(result).toEqual({ targets: [{ lang: "keras", out: "out" }] });
  });

  it("returns null for invalid JSON", () => {
    writeFileSync(resolve(TMP, ".mlmdrc"), "{ bad json }");
    const result = loadConfig(TMP);
    expect(result).toBeNull();
  });

  it("returns null for invalid structure", () => {
    writeFileSync(resolve(TMP, ".mlmdrc"), JSON.stringify({ plugins: 123 }));
    const result = loadConfig(TMP);
    expect(result).toBeNull();
  });

  it("uses MLMD_CONFIG env var", () => {
    const customPath = resolve(TMP, "custom.mlmdrc");
    writeFileSync(customPath, JSON.stringify({ plugins: "./p" }));
    process.env["MLMD_CONFIG"] = customPath;
    const result = loadConfig(TMP);
    expect(result).toEqual({ plugins: "./p" });
  });
});
