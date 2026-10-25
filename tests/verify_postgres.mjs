// npm --prefix tests ci; node tests/verify_postgres.mjs
// Real PostgreSQL engine compiled to WebAssembly; not a native server deployment test.
import { PGlite } from '@electric-sql/pglite';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const sql = await fs.readFile(path.join(root, 'sql/queries_analiticas.sql'), 'utf8');
const parts = sql.split(/^-- @section (\w+)\s*$/m);
const sections = Object.fromEntries(Array.from({length:(parts.length-1)/2}, (_,i)=>[parts[2*i+1],parts[2*i+2].trim()]));
const db = await PGlite.create();
const version = (await db.query('SELECT version() AS version')).rows[0].version;
await db.exec(sections.schema);
for (const table of ['transacoes', 'scores_modelo']) {
  const content = await fs.readFile(path.join(root, 'outputs/pg_inputs',table+'.csv'));
  await db.query(`COPY ${table} FROM '/dev/blob' WITH (FORMAT CSV, HEADER TRUE)`, [], {blob:new Blob([content])});
}
const parameters=JSON.parse(await fs.readFile(path.join(root,'outputs/pg_inputs/parameters.json'),'utf8'));
await db.query('INSERT INTO parametros_analise VALUES ($1,$2,$3,$4,$5)',parameters);
const destination=path.join(root,'outputs/postgres_results');
await fs.mkdir(destination,{recursive:true});
const rowCounts={};
for (const [name,query] of Object.entries(sections)) {
  if (name==='schema') continue;
  const clean=query.replace(/;\s*$/,'');
  const result=await db.query(`COPY (${clean}) TO '/dev/blob' WITH (FORMAT CSV, HEADER TRUE)`);
  const bytes=Buffer.from(await result.blob.arrayBuffer());
  await fs.writeFile(path.join(destination,name+'.csv'),bytes);
  rowCounts[name]=bytes.toString('utf8').trim().split(/\r?\n/).length-1;
}
await fs.writeFile(path.join(destination,'engine.json'),JSON.stringify({engine:'PostgreSQL via PGlite',version,rowCounts},null,2));
await db.close();
console.log(JSON.stringify({version,rowCounts},null,2));
