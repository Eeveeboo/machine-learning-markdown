import type { ParamValue, SourceLoc } from "./nodes.js";

/** Concrete shape dimensions, e.g. [3, 224, 224] */
export type Shape = number[];

/** A single computational node in the graph. */
export interface Block {
  /** Unique generated id, e.g. "block_0" */
  id: string;
  /** Layer type, e.g. "Conv2d" */
  type: string;
  params: Record<string, ParamValue>;
  /** Populated by shape inference */
  inputShapes: Shape[];
  /** Populated by shape inference; array supports multi-output blocks (e.g. LSTM: [hidden, cell]) */
  outputShapes: Shape[];
  /** Populated by shape inference; total parameter count for this block */
  paramCount?: number;
  /** Whether block height should scale with channel depth in SVG (from BlockDef.showDepth). Default: true. */
  showDepth?: boolean;
  loc: SourceLoc;
}

/** A directed data-flow edge between two blocks. */
export interface Edge {
  /** Block.id of the source block */
  from: string;
  /** Block.id of the destination block */
  to: string;
  /** Named tensor if this edge was created via `-> [name]` */
  tensorName?: string;
  /** Populated by shape inference */
  shape?: Shape;
}

/** A logical grouping / namespace for a set of blocks. */
export interface Group {
  /** Hierarchical path, e.g. ["ResNet50", "Bottleneck"] */
  path: string[];
  /** The ids of blocks belonging to this group */
  blockIds: string[];
}

/** The top-level directed graph IR produced after parsing. */
export interface Graph {
  blocks: Block[];
  edges: Edge[];
  groups: Group[];
}
