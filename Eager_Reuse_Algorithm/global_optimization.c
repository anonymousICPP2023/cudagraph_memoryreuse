#include "common.h"

void full_permutation(Global_opt *Opt, int level)
{
	int i, j;
	if (level >= Opt->nodenum)
	{
		for (j = 0; j < level; j++) {
			Opt->idx[Opt->count][j] = Opt->result[j];
		}
		(Opt->count)++;
		return;
	}
	for (i = 0; i < Opt->nodenum; i++)
	{
		if (Opt->flag[i] == 0)
		{
			Opt->flag[i] = 1;
			Opt->result[level++] = Opt->index[i];
			full_permutation(Opt, level);
			level--;
			Opt->flag[i] = 0;
		}
	}
}

int sum_permutation(int N)
{
	if (N == 1)
		return N;
	return sum_permutation(N - 1) * N;

}


int global_optimization(Graph *graph, int step) 
{
	Node *cmp_node, *cur_node;
	int i;
	int max = 0;

	cur_node = graph->node + *(graph->idx + step);

	for (i = 0; i < step; i++) {
		cmp_node = graph->node + *(graph->idx + i);
		if (is_lifetime_overlap(cmp_node, cur_node)) {
			if (max < cmp_node->addr + cmp_node->size) {
				max = cmp_node->addr + cmp_node->size;}
		}
	}

	cur_node->addr = max;

	return 0;
}

