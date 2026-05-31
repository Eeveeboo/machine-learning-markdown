export interface SourceLoc {
  line: number;
  col: number;
  offset: number;
}

export interface NumberVal {
  kind: "number";
  value: number;
  loc: SourceLoc;
}

export interface StringVal {
  kind: "string";
  value: string;
  loc: SourceLoc;
}

export interface BoolVal {
  kind: "bool";
  value: boolean;
  loc: SourceLoc;
}

export interface BarewordVal {
  kind: "bareword";
  value: string;
  loc: SourceLoc;
}

export interface ShapeVal {
  kind: "shape";
  dims: number[];
  loc: SourceLoc;
}

export interface ListVal {
  kind: "list";
  items: ParamValue[];
  loc: SourceLoc;
}

export type ParamValue =
  | NumberVal
  | StringVal
  | BoolVal
  | ShapeVal
  | ListVal
  | BarewordVal;

export type Param = {
  name: string;
  value: ParamValue;
  loc: SourceLoc;
};

export interface BlockDecl {
  kind: "block";
  blockType: string;
  params: Param[];
  loc: SourceLoc;
}

export interface TensorName {
  kind: "tensor_name";
  names: string[];
  loc: SourceLoc;
}

export interface TensorJoin {
  kind: "tensor_join";
  sources: string[];
  target: BlockDecl;
  loc: SourceLoc;
}

export interface GroupDecl {
  kind: "group";
  path: string[];
  body: ASTNode[];
  loc: SourceLoc;
}

export interface Comment {
  kind: "comment";
  text: string;
  loc: SourceLoc;
}

export type ASTNode = BlockDecl | TensorName | TensorJoin | GroupDecl | Comment;
