## HTCondor

To see how many nodes are available, use

    condor_status

## PySTAMPAS

### 1 - Install and initiate PySTAMPAS
We will work on a custom branch of PySTAMPAS that is slightly different from the main branch, so that we can do modifications of the code without affecting the pipeline used for LVK analyses.

* Clone the `pystampas` repository into your `/home` and switch to the branch "statistic"
 ```
 git clone https://git.ligo.org/adrian.macquet/pystampas.git
 git checkout -b statistic
 git pull origin statistic
   ```



###  2 - Set up a new project

Create a new project directory (e.g "myProject") with

    stampas --new_project myProject

Go to the project's directory. It contains several folders and files used to configure the analysis. 

### 3 - Configure parameter files
Parameter files tell the pipeline what to do, which data to process and how. They are stored in  `params/`. There are several parameter files corresponding to different aspects of the pipeline. When a project is created, these files contain default values, so they need to be customized for each search.

You can copy the following params/ folder into your project to have default params adapted for the ET MDC configuration.

    cp -r /home/adrian-macquet/Projects/ET_MDC_1v2/pystampas_analyzes/day18_3dets/params ./


First, we want to estimate the background, i.e the rate of noise triggers. The relevant parameter file for that is `bkg.yml`. Several fields must be customized:

* The most important are START and STOP, to define the segment of data to analyze. The MDC dataset starts at GPS time 1000000000 and ends 1 month later. The default values represent a segment of 1 day of data (the 18th day of the MDC) that contains the loudest BBH.
* All the paths to the current project defined in the parameter files must be changed. Search and replace  `/home/adrian-macquet/Projects/ET_MDC_1v2/pystampas_analyzes/day18_3dets/` with the path to your own project directory.

### 4 - Configure submission files
Submission files are located in dag/ and end in .sub. They manage the Condor workflow. You need to open the following ones:
* pystampas_stage1.sub
* pystampas_stage2.sub
* pystampas_stage2_zerolag.sub

And for each file, in the fields output and error, replace the path to the default directory `/home/adrian-macquet/Projects/ET_MDC_1v2/pystampas_analyzes/day18_3dets/` with your own project directory.

### 5 - Run stage 1

Set up the workflow (dag) for stage 1 with:

    stampas --dag_stage1 --paramsFile /home/..../your_project/params/bkg.yml

(you need to give the absolute path to the bkg.yml parameter file).

Submit the workflow to Condor with

    condor_submit_dag -maxjobs 40 dag/BKG/pystampas_stage1.dag

You can monitor the progression of the workflow with

    tail -f dag/BKG/pystampas_stage1.dag.dagman.out

When all jobs have completed, run the post-processing script for stage1:

    python postprocessing/run_stage1_postprocessing params/bkg.yml

(note: it is possible that the last job fails. It is not a problem and you can still run the postprocessing script).

### Run stage 2 for background estimation

Set up the workflow (dag) for stage 2 with:

    stampas --dag_stage2 --paramsFile /home/..../your_project/params/bkg.yml

Submit the workflow to Condor with

    condor_submit_dag -maxjobs 40 dag/BKG/pystampas_stage2.dag

When all jobs have completed, run the post-processing script for stage:

    python postprocessing/run_background_postprocessing params/bkg.yml

