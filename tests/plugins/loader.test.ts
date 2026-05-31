import { describe, it, expect, beforeEach, afterEach } from "vitest";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { loadPlugins } from "../../src/plugins/loader.js";
import { registry } from "../../src/blocks/registry.js";

let tmpDir: string;

beforeEach(() => {
  tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "nnml-plugin-test-"));
  registry.clear();
});

afterEach(() => {
  fs.rmSync(tmpDir, { recursive: true, force: true });
});

describe("loadPlugins", () => {
  it("loads a .mjs plugin and returns it in the map", async () => {
    const pluginSrc = `
export default {
  inputs: ["x"],
  outputs: ["y"],
  inferShape(inputs, _params) { return inputs; },
};
`;
    fs.writeFileSync(path.join(tmpDir, "MyLayer.mjs"), pluginSrc, "utf8");

    const plugins = await loadPlugins(tmpDir);

    expect(plugins.has("MyLayer")).toBe(true);
    const plugin = plugins.get("MyLayer")!;
    expect(plugin.inputs).toEqual(["x"]);
    expect(plugin.outputs).toEqual(["y"]);
  });

  it("registers plugin with inferShape into block registry", async () => {
    const pluginSrc = `
export default {
  inputs: ["x"],
  outputs: ["y"],
  inferShape(inputs, _params) { return [[42]]; },
};
`;
    fs.writeFileSync(path.join(tmpDir, "CustomBlock.mjs"), pluginSrc, "utf8");

    await loadPlugins(tmpDir);

    const def = registry.get("CustomBlock");
    expect(def).toBeDefined();
    expect(def!.name).toBe("CustomBlock");
    const shape = def!.inferShape([[1, 2]], {});
    expect(shape).toEqual([[42]]);
  });

  it("accepts named export matching the filename", async () => {
    const pluginSrc = `
export const NamedPlugin = {
  inputs: ["a"],
  outputs: ["b"],
};
`;
    fs.writeFileSync(path.join(tmpDir, "NamedPlugin.mjs"), pluginSrc, "utf8");

    const plugins = await loadPlugins(tmpDir);
    expect(plugins.has("NamedPlugin")).toBe(true);
  });

  it("skips files that do not export a valid BlockPlugin", async () => {
    fs.writeFileSync(path.join(tmpDir, "NotAPlugin.mjs"), `export default { foo: "bar" };`, "utf8");

    const plugins = await loadPlugins(tmpDir);
    expect(plugins.has("NotAPlugin")).toBe(false);
  });

  it("prefers .mjs over .js over .ts for the same base name", async () => {
    const mjsSrc = `export default { inputs: ["mjs"], outputs: [] };`;
    const jsSrc  = `export default { inputs: ["js"],  outputs: [] };`;
    fs.writeFileSync(path.join(tmpDir, "Dup.mjs"), mjsSrc, "utf8");
    fs.writeFileSync(path.join(tmpDir, "Dup.js"),  jsSrc,  "utf8");

    const plugins = await loadPlugins(tmpDir);
    expect(plugins.get("Dup")?.inputs).toEqual(["mjs"]);
  });

  it("returns empty map for non-existent directory", async () => {
    const plugins = await loadPlugins(path.join(tmpDir, "no-such-dir"));
    expect(plugins.size).toBe(0);
  });

  it("does not register plugins without inferShape into block registry", async () => {
    const pluginSrc = `export default { inputs: ["x"], outputs: ["y"] };`;
    fs.writeFileSync(path.join(tmpDir, "NoShape.mjs"), pluginSrc, "utf8");

    await loadPlugins(tmpDir);
    expect(registry.has("NoShape")).toBe(false);
  });
});
