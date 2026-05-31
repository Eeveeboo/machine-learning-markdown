export type { ParamType, ParamSpec, BlockDef } from "./types.js";
export { registry, registerBlock, lookupBlock } from "./registry.js";
export { registerBuiltins } from "../plugins/builtins/index.js";

import "../plugins/builtins/index.js";
