import helper, { twice as double } from "./helper.js";

export function run(value) {
  return double(helper(value));
}
