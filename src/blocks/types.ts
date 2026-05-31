import type { Shape } from "../ast/graph.js";
import type { ParamValue } from "../ast/nodes.js";

export type ParamType = "number" | "string" | "bool" | "shape" | "list" | "bareword";

export interface ParamSpec {
  name: string;
  type: ParamType;
  required: boolean;
  default?: ParamValue;
}

export interface BlockDef {
  name: string;
  params: ParamSpec[];
  inferShape(inputs: Shape[], params: Record<string, ParamValue>): Shape[];
  paramCount?(inputs: Shape[], params: Record<string, ParamValue>): number;
  /** If false, block height won't scale with channel depth in SVG layout.
   *  Set for passthrough blocks (activations, merges, etc.) where depth
   *  doesn't represent meaningful tensor transformation. Default: true. */
  showDepth?: boolean;
}
