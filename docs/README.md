# OpenAssembler documentation

Choose a path by what you want to do. The established command, workflow, status,
compiler, research and reverse-engineering records remain detailed references;
this page links to them rather than copying their evidence tables.

## Getting started

- [Install OpenAssembler](getting-started/installation.md)
- [Compile and run a typed Python kernel](getting-started/first-kernel.md)
- [Compile a native Python model](getting-started/first-model.md)
- [Use Modal or your own A100](getting-started/modal-setup.md)

## Apply it to a project

- [Supported inputs and limits](user-guide/supported-inputs.md)
- [Discover a model and choose its inputs](REPOSITORY_WORKFLOW.md#automatic-model-discovery)
- [Configure, compile, run and inspect artifacts](REPOSITORY_WORKFLOW.md)
- [Write custom typed Python kernels](FRONTENDS.md)
- [Describe a kernel launch](COMMANDS.md#describe-an-actual-launch-with---spec)
- [Troubleshoot builds and execution](user-guide/troubleshooting.md)

## Look up an exact contract

- [CLI syntax and options](COMMANDS.md)
- [Model operation support](reference/supported-operators.md)
- [Model configuration](reference/configuration.md)
- [Current architecture and status](STATUS.md)
- [Source and execution routes](architecture/compilation-pipelines.md)
- [Host and toolkit dependencies](architecture/dependencies.md)

## Compiler, evidence and development

- [Compiler contracts and known gaps](compiler/README.md)
- [Feature research catalogue](research/README.md)
- [Reverse-engineering results](reverse_eng_results/README.md)
- [Reproducible evidence archive](evidence/)
- [Contributor setup and validation](DEVELOPMENT.md)
- [Documentation maintenance rules](development/documentation-guidelines.md)

Existing command, frontend, development and workflow references remain in
place so their links keep working. Current implementation contracts live in
[compiler/](compiler/README.md); feature studies are indexed in
[research/](research/README.md); measured records are organized in
[evidence/](evidence/). Historical experiments under the repository's exp
directory are linked from the records that explain their scope.

If an introductory page and an implementation dossier disagree, trust current
source and the linked contract. Update the user-facing route and detailed
contract together when fixing a discrepancy.
