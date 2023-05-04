#include "common.h"

int eager_reuse_test(Graph graph, FILE *fp){
	enum algorithmType type=eager_reuse_type;
	struct timeval begin, end;
	struct timeval tv[2];
	Three_layers *structure, *temp, *front;
	Node *first_node, *middle_node, *last_node;
	int min = 0;
	int max = 0;

	structure = (Three_layers *)malloc(sizeof(Three_layers));
	structure->next = NULL;
	find_structure(graph, structure);
	front = structure;
	temp = structure->next;

	gettimeofday(&tv[0], NULL);
	while(temp) {
		first_node = graph.node + temp->first - 1;
		middle_node = graph.node + temp->middle - 1;
		last_node = graph.node + temp->last - 1;
		if (!(first_node->done * middle_node->done * last_node->done)
				&& determine_execution_order(&graph, &temp)){
			first_node = graph.node + temp->first - 1;
			middle_node = graph.node + temp->middle - 1;
			last_node = graph.node + temp->last - 1;

			eager_reuse_for_one_threelayer_structure(&graph, temp);
			if (min > first_node->addr) {
				min = first_node->addr;
			}

			if (min > middle_node->addr) {
				min = middle_node->addr;
			}
			if (min > last_node->addr) {
				min = last_node->addr;
			}

			if (max < first_node->addr + first_node->size){
				max = first_node->addr + first_node->size;
			}

			if (max < middle_node->addr + middle_node->size){
				max = middle_node->addr + middle_node->size;
			}

			if (max < last_node->addr + last_node->size){
				max = last_node->addr + last_node->size;
			}
		}

		front->next = temp;
		front = temp;
		temp = temp->next;
	}
	gettimeofday(&tv[1], NULL);

	dump_result(fp, &graph, tv, max, min, type);

	return 0;
}


int large_tensor_first_v1_test(Graph graph, FILE *fp){
	enum algorithmType type=large_tensor_first_v1_type;
	int memory_size = 0;
	int this_top;
	struct timeval tv[2];
	Node cur_node;
	int i, ret = 0;

	sort_large_first(&graph);

	gettimeofday(&tv[0], NULL);
	for (i = 0; i < graph.nodenum; i++){
		ret = large_tensor_first_v1(&graph, i);
		if (ret)
			return 1;

		cur_node = *(graph.node + *(graph.idx + i));
		this_top = cur_node.addr + cur_node.size;

		if (memory_size < this_top){
			memory_size = this_top;
		}
	}
	gettimeofday(&tv[1], NULL);

	dump_result(fp, &graph, tv, memory_size, 0, type);

	return 0;
}

int large_tensor_first_v2_test(Graph graph, FILE *fp)
{
	enum algorithmType type=large_tensor_first_v2_type;
	int memory_size = 0;
	int this_top;
	struct timeval tv[2];
	int i, ret = 0;
	Node cur_node;

	sort_large_first(&graph);
	gettimeofday(&tv[0], NULL);
	for (i = 0; i < graph.nodenum; i++){
		ret = large_tensor_first_v2(&graph, i);
		if (ret)
			return 1;

		cur_node = *(graph.node + *(graph.idx + i));
		this_top = cur_node.addr + cur_node.size;
		if (memory_size < this_top){
			memory_size = this_top;
		}
	}

	gettimeofday(&tv[1], NULL);

	dump_result(fp, &graph, tv, memory_size, 0, type);

	return 0;
}


int short_lifetime_first_test(Graph graph, FILE *fp)
{
	enum algorithmType type=short_lifetime_first_type;
	int memory_size = 0;
	int this_top;
	int i;
	int ret = 0;
	Node cur_node;
	struct timeval tv[2];

	sort_short_lifetime(&graph);

	gettimeofday(&tv[0], NULL);
	for (i = 0; i < graph.nodenum; i++) {
		ret = short_lifetime_first(&graph, i);
		if (ret) 
			return 1;

		cur_node = *(graph.node + *(graph.idx + i));
		this_top = cur_node.addr + cur_node.size;

		if (memory_size < this_top){
			memory_size = this_top;
		}
	}
	gettimeofday(&tv[1], NULL);

	dump_result(fp, &graph, tv, memory_size, 0, type);

	return 0;
}

int main(int argc,char** argv) 
{
	char *input_file = (char *)calloc(1, 256);
	char *result_file = (char *)calloc(1, 256);
	Graph graph;
	FILE *fp;
	int ret = 0;

	if (argc < 2){
		printf("no input parameter!\n");
		exit(0);
	} else{
		sprintf(input_file, "input/input%s", argv[1]);
		sprintf(result_file, "result/result%s", argv[1]);
		if ((fp = fopen(result_file, "w")) == NULL){
			printf("open file error!\n");
			exit(0);
		}
	}
	createGraphByFile(&graph, input_file);

	output_graph2file(graph, fp);

	ret = eager_reuse_test(graph, fp);
	if (ret)
		printf("eager reuse test failed!!!\n");

	/* v2 is better */
	ret = large_tensor_first_v2_test(graph, fp);
	if (ret)
		printf("large tensor first v2 test failed!!!\n");

	/* v1 is worse 
	   ret = large_tensor_first_v1_test(graph, fp);
	   if (ret)
	   printf("large tensor first v1 test failed!!!\n");
	   */

	ret = short_lifetime_first_test(graph, fp);
	if (ret)
		printf("short lifetime first test failed!!!\n");

	fclose(fp);

    destoryGraph(&graph);

	return 0;
}
