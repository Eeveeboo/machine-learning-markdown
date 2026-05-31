import type { Block } from '../ast/graph.js';
import type { SvgBuilder } from './svg-builder.js';

export interface RenderContext {
  svg: SvgBuilder;
  block: Block;
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface CodegenContext {
  block: Block;
  inputs: string[];
  outputVar: string;
  targetLang: string;
}
