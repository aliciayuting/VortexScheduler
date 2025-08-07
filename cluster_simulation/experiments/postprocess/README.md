To generate an HTML comparison summary table based on the `stats.json` file from simulator runs, provide a CSV file with headers `group,scheduler,path_to_data` with the name of the table group for each table you would like to generate, the scheduler type name, and the path to the results of the experiment for that scheduler.  
  
**Example:**
```
group,scheduler,path_to_data
No SLO Send Rate 55,Hash task (HERD),../auto_script/results/hash-herd-p1-55-4node-noslo/hashtask
No SLO Send Rate 55,Shepherd (HERD),../auto_script/results/shep-herd-p1-55-4node-noslo/shepherd
No SLO Send Rate 55,Decentral HEFT (HERD),../auto_script/results/dch-herd-p1-55-4node-noslo/decentralheft
No SLO Send Rate 55,Hash task (MIG),../auto_script/results/hash-mig-p1-55-4node-noslo/hashtask
No SLO Send Rate 55,Shepherd (MIG),../auto_script/results/shep-mig-p1-55-4node-noslo/shepherd
No SLO Send Rate 55,Decentral HEFT (MIG),../auto_script/results/dch-mig-p1-55-4node-noslo/decentralheft
Job-Level SLO=303ms 0 Tardiness Send Rate 55,Hash task (HERD),../auto_script/results/hash-herd-p1-55-4node-0slack-303slo/hashtask
Job-Level SLO=303ms 0 Tardiness Send Rate 55,Shepherd (HERD),../auto_script/results/shep-herd-p1-55-4node-0slack-303slo/shepherd
Job-Level SLO=303ms 0 Tardiness Send Rate 55,Decentral HEFT (HERD),../auto_script/results/dch-herd-p1-55-4node-0slack-303slo/decentralheft
Job-Level SLO=303ms 0 Tardiness Send Rate 55,Hash task (MIG),../auto_script/results/hash-mig-p1-55-4node-0slack-303slo/hashtask
Job-Level SLO=303ms 0 Tardiness Send Rate 55,Shepherd (MIG),../auto_script/results/shep-mig-p1-55-4node-0slack-303slo/shepherd
```