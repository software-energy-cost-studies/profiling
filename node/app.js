
const { InfluxDB, Point } = require("@influxdata/influxdb-client");
// You can generate a Token from the "Tokens Tab" in the UI
const token =
  "YOUR_TOKEN";
const org = "YOUR_ORG";
const bucket = "YOUR_BUCKET";
const client = new InfluxDB({ url: "http://localhost:8086", token: token });
const writeApi = client.getWriteApi(org, bucket);
writeApi.useDefaultTags({ host: "host1" });
const spawn = require("child_process").spawn;
const power = spawn("powermetrics", [
  "-i 1000 --samplers cpu_power,gpu_power -a --hide-cpu-duty-cycle --show-usage-summary --show-extra-power-info",
]);
power.stdout.on("data", (data) => {
  console.log(Date.now(), " - Signal captured.");
  const strings = data.toString().split(/\n/);
  const extract = strings.filter((s) => {
    return s.includes("mW");
  });
extract.forEach((data) => {
    const kv = data.split(" Power: ");
    const [k, v] = kv;
    if (k && v) {
      const point = new Point(k).uintField(k, parseInt(v.replace("mW")));
      writeApi.writePoint(point);
console.log(k, v)
    }
  });
});


