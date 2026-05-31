// mlmd - Neural network architecture DSL

// Parser
export { tokenize } from "./parser/tokenizer.js";
export type { Token, TokenType } from "./parser/tokenizer.js";
export { parse, buildGraph } from "./parser/index.js";
export type { ParseResult, ParseError } from "./parser/index.js";

// AST & Graph
export type {
  SourceLoc,
  ParamValue,
  Param,
  BlockDecl,
  TensorName,
  TensorJoin,
  GroupDecl,
  Comment,
  ASTNode,
} from "./ast/nodes.js";
export type { Graph, Block, Edge, Group, Shape } from "./ast/graph.js";
export { topoSort, topoSortIds, buildAdjacency } from "./ast/graph-utils.js";
export type { NodeInfo } from "./ast/graph-utils.js";

// Block registry
export { registerBlock, lookupBlock, registry } from "./blocks/registry.js";
export type { BlockDef, ParamSpec, ParamType } from "./blocks/types.js";

// Shape inference
export { inferShapes } from "./shape/infer.js";
export type { ShapeResult, ShapeError } from "./shape/infer.js";

// Lint
export { lint } from "./lint/index.js";
export type { LintDiagnostic } from "./lint/index.js";

// Codegen
export { getTarget, registerTarget } from "./codegen/target.js";
export type { CodegenTarget, GeneratedFile } from "./codegen/target.js";

// Visualize
export { layout, render } from "./visualize/index.js";
export type { LayoutResult, LayoutNode, LayoutEdge } from "./visualize/layout.js";

// Config
export { loadConfig } from "./config/loader.js";
export type { MlmdConfig } from "./config/schema.js";

// Plugins
export { loadPlugins } from "./plugins/loader.js";
export type { BlockPlugin } from "./plugins/types.js";
