import type { Graph } from "../ast/graph.js";
import type { BlockDef } from "../blocks/types.js";

export interface GeneratedFile {
  path: string;
  content: string;
}

export interface CodegenTarget {
  name: string;
  fileExtension: string;
  generate(graph: Graph, registry: Map<string, BlockDef>): GeneratedFile[];
}

const registry = new Map<string, CodegenTarget>();

export function getTarget(name: string): CodegenTarget | undefined {
  return registry.get(name);
}

export function registerTarget(target: CodegenTarget): void {
  registry.set(target.name, target);
}
