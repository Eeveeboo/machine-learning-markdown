import * as vscode from 'vscode';
import {
  LanguageClient,
  LanguageClientOptions,
  ServerOptions,
  TransportKind,
} from 'vscode-languageclient/node';

let client: LanguageClient;

export function activate(context: vscode.ExtensionContext): void {
  const config = vscode.workspace.getConfiguration('mlmd');
  const serverPath: string = config.get('serverPath') ?? 'mlmd';

  const serverOptions: ServerOptions = {
    command: serverPath,
    args: ['lsp'],
    transport: TransportKind.stdio,
  };

  const clientOptions: LanguageClientOptions = {
    documentSelector: [{ scheme: 'file', language: 'mlmd' }],
    synchronize: {
      fileEvents: vscode.workspace.createFileSystemWatcher('**/*.mlmd'),
    },
  };

  client = new LanguageClient(
    'mlmd',
    'MLMD Language Server',
    serverOptions,
    clientOptions,
  );

  client.start();

  context.subscriptions.push(
    vscode.commands.registerCommand('mlmd.visualize', async () => {
      const editor = vscode.window.activeTextEditor;
      if (!editor) {
        vscode.window.showWarningMessage('No active MLMD file to visualize.');
        return;
      }
      const uri = editor.document.uri.fsPath;
      const terminal = vscode.window.createTerminal('MLMD Visualize');
      terminal.show();
      terminal.sendText(`${serverPath} visualize "${uri}"`);
    }),

    vscode.commands.registerCommand('mlmd.generate', async () => {
      const editor = vscode.window.activeTextEditor;
      if (!editor) {
        vscode.window.showWarningMessage('No active MLMD file to generate from.');
        return;
      }
      const target: string = config.get('defaultTarget') ?? 'pytorch';
      const uri = editor.document.uri.fsPath;
      const terminal = vscode.window.createTerminal('MLMD Generate');
      terminal.show();
      terminal.sendText(`${serverPath} generate --target ${target} "${uri}"`);
    }),

    vscode.commands.registerCommand('mlmd.lint', async () => {
      const editor = vscode.window.activeTextEditor;
      if (!editor) {
        vscode.window.showWarningMessage('No active MLMD file to lint.');
        return;
      }
      const uri = editor.document.uri.fsPath;
      const terminal = vscode.window.createTerminal('MLMD Lint');
      terminal.show();
      terminal.sendText(`${serverPath} lint "${uri}"`);
    }),
  );
}

export function deactivate(): Promise<void> | undefined {
  return client?.stop();
}
