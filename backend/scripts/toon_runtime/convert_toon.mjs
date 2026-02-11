#!/usr/bin/env node

import { readFileSync, writeFileSync } from "node:fs";
import process from "node:process";
import * as toon from "@toon-format/toon";

function parseArgs(argv) {
  const args = { input: null, output: null };
  for (let i = 0; i < argv.length; i += 1) {
    const token = argv[i];
    if (token === "--input" && argv[i + 1]) {
      args.input = argv[i + 1];
      i += 1;
    } else if (token === "--output" && argv[i + 1]) {
      args.output = argv[i + 1];
      i += 1;
    }
  }
  return args;
}

function toToonString(value) {
  if (typeof value === "string") {
    return value;
  }
  if (value instanceof Uint8Array) {
    return Buffer.from(value).toString("utf8");
  }
  if (value && typeof value === "object") {
    if (typeof value.toon === "string") {
      return value.toon;
    }
    if (typeof value.output === "string") {
      return value.output;
    }
  }
  throw new Error("converter returned unsupported output type");
}

function collectCandidates(moduleRef) {
  const names = [
    "toTOON",
    "toToon",
    "fromJSON",
    "fromJson",
    "convert",
    "encode",
    "serialize",
    "stringify",
  ];
  const candidates = [];

  for (const name of names) {
    if (typeof moduleRef[name] === "function") {
      candidates.push({ name, fn: moduleRef[name] });
    }
  }

  if (typeof moduleRef.default === "function") {
    candidates.push({ name: "default", fn: moduleRef.default });
  } else if (moduleRef.default && typeof moduleRef.default === "object") {
    for (const name of names) {
      if (typeof moduleRef.default[name] === "function") {
        candidates.push({ name: `default.${name}`, fn: moduleRef.default[name] });
      }
    }
  }
  return candidates;
}

async function convertPayload(payload) {
  const candidates = collectCandidates(toon);
  const errors = [];

  for (const candidate of candidates) {
    try {
      const maybeResult = await Promise.resolve(candidate.fn(payload));
      return toToonString(maybeResult);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      errors.push(`${candidate.name}: ${message}`);
    }
  }

  const exportNames = Object.keys(toon);
  const details = errors.length > 0 ? errors.join(" | ") : "no callable converter export";
  throw new Error(
    `Unable to convert JSON payload with @toon-format/toon (exports: ${exportNames.join(", ")}). ${details}`,
  );
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const inputText = args.input
    ? readFileSync(args.input, "utf8")
    : readFileSync(0, "utf8");
  const payload = JSON.parse(inputText);
  const toonText = await convertPayload(payload);

  if (!toonText || toonText.trim().length === 0) {
    throw new Error("TOON conversion produced empty output");
  }

  if (args.output) {
    writeFileSync(args.output, toonText, "utf8");
  } else {
    process.stdout.write(toonText);
  }
}

try {
  await main();
} catch (error) {
  const message = error instanceof Error ? error.message : String(error);
  process.stderr.write(`convert_toon failed: ${message}\n`);
  process.exit(1);
}
