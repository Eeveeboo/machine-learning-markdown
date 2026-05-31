import type { ASTNode, BlockDecl, Param } from "../ast/nodes.js";
import type { Block, Edge, Graph, Group } from "../ast/graph.js";
import type { ParamValue } from "../ast/nodes.js";

function paramsToRecord(params: Param[]): Record<string, ParamValue> {
  const result: Record<string, ParamValue> = {};
  for (const p of params) {
    result[p.name] = p.value;
  }
  return result;
}

interface BuildState {
  blocks: Block[];
  edges: Edge[];
  groups: Group[];
  // count per blockType for unique ID generation
  typeCount: Map<string, number>;
  // named tensors: name -> block id
  namedTensors: Map<string, string>;
  // current tail block id (last block in linear chain)
  currentTail: string | null;
}

function makeBlockId(state: BuildState, blockType: string): string {
  const count = state.typeCount.get(blockType) ?? 0;
  state.typeCount.set(blockType, count + 1);
  return `${blockType}_${count}`;
}

function addBlock(state: BuildState, decl: BlockDecl): Block {
  const id = makeBlockId(state, decl.blockType);
  const block: Block = {
    id,
    type: decl.blockType,
    params: paramsToRecord(decl.params),
    inputShapes: [],
    outputShapes: [],
    loc: decl.loc,
  };
  state.blocks.push(block);
  return block;
}

function processNodes(
  state: BuildState,
  nodes: ASTNode[],
  groupStack: string[][]
): void {
  for (const node of nodes) {
    switch (node.kind) {
      case "comment":
        // skip
        break;

      case "block": {
        const block = addBlock(state, node);
        // Wire edge from current tail
        if (state.currentTail !== null) {
          state.edges.push({ from: state.currentTail, to: block.id });
        }
        state.currentTail = block.id;
        // Register block in all ancestor groups
        for (const grp of groupStack) {
          grp.push(block.id);
        }
        break;
      }

      case "tensor_name": {
        // Name the current tail for each name provided
        if (state.currentTail !== null) {
          for (const name of node.names) {
            state.namedTensors.set(name, state.currentTail);
          }
        }
        break;
      }

      case "tensor_join": {
        // Create a new block from the target BlockDecl
        const block = addBlock(state, node.target);
        // Wire edges from each named source
        for (const src of node.sources) {
          const fromId = state.namedTensors.get(src);
          if (fromId !== undefined) {
            state.edges.push({ from: fromId, to: block.id, tensorName: src });
          }
        }
        state.currentTail = block.id;
        // Register block in all ancestor groups
        for (const grp of groupStack) {
          grp.push(block.id);
        }
        break;
      }

      case "group": {
        // Create a new group and recurse
        const blockIds: string[] = [];
        const group: Group = { path: node.path, blockIds };
        state.groups.push(group);
        processNodes(state, node.body, [...groupStack, blockIds]);
        break;
      }
    }
  }
}

export function buildGraph(nodes: ASTNode[]): Graph {
  const state: BuildState = {
    blocks: [],
    edges: [],
    groups: [],
    typeCount: new Map(),
    namedTensors: new Map(),
    currentTail: null,
  };
  processNodes(state, nodes, []);
  // Cleanup: remove spurious chain edges between Input blocks
  state.edges = state.edges.filter(e => {
    const fromBlock = state.blocks.find(b => b.id === e.from);
    const toBlock = state.blocks.find(b => b.id === e.to);
    if (fromBlock && toBlock && fromBlock.type === "Input" && toBlock.type === "Input") {
      return false;
    }
    return true;
  });
  return { blocks: state.blocks, edges: state.edges, groups: state.groups };
}
