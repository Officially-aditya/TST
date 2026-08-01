import Widget, { helper as doHelp } from "./helper";
import * as tools from "./util.ts";

interface Runnable {
  run(): string;
}

export class Service extends Widget implements Runnable {
  run(): string {
    return tools.format(doHelp());
  }
}
