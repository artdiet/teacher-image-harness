#!/usr/bin/env node
// Teach Image MCP server (stdio). Claude Desktop starts it on the computer where Claude runs;
// it forwards tool calls to the image server on the network. Logs go to stderr only:
// stdout carries the MCP protocol.

import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";

import { createClient } from "./client.js";
import { loadConfig } from "./config.js";
import { VERSION, createServer } from "./server.js";

const config = loadConfig();
const server = createServer(config, createClient(config));
await server.connect(new StdioServerTransport());
console.error(`[teach-image] ${VERSION} ready; image server ${config.serverUrl}, ` +
  `token ${config.token ? "set" : "missing"}, pictures saved in ${config.outputDir}` +
  (config.problem ? `; settings problem: ${config.problem}` : ""));
