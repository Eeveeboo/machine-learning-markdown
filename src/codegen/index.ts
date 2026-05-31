export type { GeneratedFile, CodegenTarget } from "./target.js";
export { getTarget, registerTarget } from "./target.js";

// Register built-in targets (side-effect imports trigger registerTarget)
import "./targets/pytorch.js";
import "./targets/keras.js";
import "./targets/candle.js";
