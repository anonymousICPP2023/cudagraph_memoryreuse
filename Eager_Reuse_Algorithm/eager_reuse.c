#include "common.h"

void find_structure(Graph graph, Three_layers *s) {
	int i, j, k, first, middle, last;
	Three_layers *end = s;
	Three_layers *data;
	Node *first_node, *middle_node;

	for(i = 0; i < graph.nodenum; i++) {
		first = (*(graph.node + i)).ID;
		first_node = graph.node + first - 1;
		for(j = 0; j < first_node->adj_num; j++) {
			middle = *(first_node->adj_idx + j);
			middle_node = graph.node + middle - 1;
			for(k = 0; k < middle_node->adj_num; k++) {
				last = *(middle_node->adj_idx + k);
				data = (Three_layers *)malloc(sizeof(Three_layers));
				data->first = first;
				data->middle = middle;
				data->last = last;
				end->next = data;
				end = data;
			}
		}
	}
	end->next = NULL;
}

static void reduce_adj_node_indegree(Graph *graph, Node *three_layer_node){
	int node_idx;

	for (int i = 0; i < three_layer_node->adj_num; i++){
		node_idx = *(three_layer_node->adj_idx + i) - 1;
		((*(graph->node + node_idx)).indegree)--;
	}
}

/* move the i_th three-layer structure to the head of link-list */
static int swap_ith2head(Three_layers **temp, Three_layers **front, Three_layers **s, int idx){
	int i;

	if (idx == 1){
		*front = (*temp)->next;
		(*s)->next = (*front)->next;
		(*front)->next = *s;
		*s = *front;
		*temp = *s;
	} else {
		for(i = 0; i < idx; i++){
			*temp = (*temp)->next;
			if (i == idx - 2) {
				*front = *temp;
			}
		}
		if (*temp == NULL) {
			return 1;
		} else {
			(*front)->next = (*temp)->next;
			(*temp)->next = *s;
			*s = *temp;
		}
	}
	return 0;
}

int determine_execution_order(Graph *graph, Three_layers **s) {
	int i = 1, j = 0;
	int node_idx;
	Three_layers *t = *s;
	Three_layers *front;
	Node *first_node, *middle_node, *last_node;

	while(1) {
		first_node = graph->node + t->first - 1;
		middle_node = graph->node + t->middle - 1;
		last_node = graph->node + t->last - 1;

		reduce_adj_node_indegree(graph,first_node);
		reduce_adj_node_indegree(graph,middle_node);
		reduce_adj_node_indegree(graph,last_node);

		if (first_node->indegree <= 0 && middle_node->indegree <= 0
				&& last_node->indegree <= 0){
			return 1;
		} else {
			if(swap_ith2head(&t, &front, s, i)){
				break;
			}
			i++;
		}
		if (i > 20){
			printf("something wrong happen\n");
			break;
		}
	}
	return 0;
}

int find_bottom(Graph *graph, int address, int idx){
	int i;
	int temp = address;
	int addr_plus_size; 
	int diff = MAX;
	Node *cmp_node;
	Node *cur_node = graph->node + idx;
	address = address + (*(graph->node + idx)).size;

	for (i = 0; i < graph->nodenum; i++){
		cmp_node = graph->node + i;

		if (cmp_node->done) {
			if (is_lifetime_overlap(cmp_node, cur_node)) {
				addr_plus_size = cmp_node->addr + cmp_node->size;

				if (address > addr_plus_size
						&& diff > address - addr_plus_size) {
					diff = address - addr_plus_size;
					temp = addr_plus_size;
				}
			}
		}
	}

	return temp;
}

void adjust_addr(Graph *graph, int address, int shift, int idx){
	int i;
	Node *cmp_node;
	Node *cur_node = graph->node + idx;

	for (i = 0; i < graph->nodenum; i++){
		cmp_node = graph->node + i;
		if (cmp_node->done) {
			if (is_lifetime_overlap(cmp_node, cur_node)) {
				if (address < cmp_node->addr){
					cmp_node->addr += shift;
				}
			}
		}
	}
	return;
}

int solve_conflict(int address, Graph *graph, int idx, int direction){
	int i;
	int found_flag = 0;
	int min = MAX;
	int bottom;
	int addr_plus_size;
	Node *cmp_node;
	Node *cur_node = graph->node + idx;

	if (direction == 1) {
		for (i = 0; i < graph->nodenum; i++) {
			cmp_node = graph->node + i;
			if (cmp_node->done) {
				if (is_lifetime_overlap(cmp_node, cur_node)) {
					addr_plus_size = cmp_node->addr + cmp_node->size;
					if (address < addr_plus_size){
						address = addr_plus_size;
					}
				}
			}
		}
		return address;
	}

	if (direction == -1) {
		for (i = 0; i < graph->nodenum; i++){
			cmp_node = graph->node + i;
			if (cmp_node->done) {
				if (is_lifetime_overlap(cmp_node, cur_node)) {
					addr_plus_size = cmp_node->addr + cmp_node->size;
					int tmp = address + cur_node->size - 1;
					if (!(tmp >= addr_plus_size || tmp < cmp_node->addr)) {
						found_flag = 1;
					}
					if (min > cmp_node->addr){
						min = cmp_node->addr;	
					}
				}
			}
		}

		if (found_flag) {
			return min - cur_node->size;
		}

		bottom = find_bottom(graph, address, idx);
		if (address >= bottom) {
			return address;
		} 

		adjust_addr(graph, bottom, bottom - address, idx);

		return bottom;
	}
	return 0;
}


void eager_reuse_for_one_threelayer_structure(Graph *graph, Three_layers *s){
	Node *node1, *node2;
	Node *first_node, *middle_node, *last_node;
	int t[4];
	int temp_addr;
	int i;

	first_node = graph->node + s->first - 1;
	middle_node = graph->node + s->middle - 1;
	last_node = graph->node + s->last - 1;

	if (first_node->tag != 0){
		middle_node->tag = -1 * first_node->tag;
		last_node->tag = first_node->tag;
	} else if (middle_node->tag != 0) {
		first_node->tag = -1 * middle_node->tag;
		last_node->tag = -1 * middle_node->tag;
	} else if (last_node->tag != 0){
		first_node->tag = last_node->tag;
		middle_node->tag = -1 * last_node->tag;
	} else {
		first_node->tag = 1;
		middle_node->tag = -1;
		last_node->tag = 1;
		first_node->addr = 0;
		first_node->done = 1;
	}

	if (first_node->done){
		t[0]= s->middle - 1;
		t[1] = s->first - 1;
		t[2] = s->last - 1;
		t[3] = s->middle - 1;
	} else if (middle_node->done){
		t[0] = s->first - 1;
		t[1] = s->middle - 1;
		t[2] = s->last - 1;	
		t[3] = s->middle - 1;
	} else if (last_node->done){
		t[0] = s->middle - 1;
		t[1] = s->last - 1;
		t[2] = s->first - 1;	
		t[3] = s->middle - 1;
	} else {
		printf("something wrong happen\n");
		return;
	}

	for (i = 0; i < 4; i += 2){
		node1 = graph->node + t[i];
		node2 = graph->node + t[i + 1];

		if (node1->done == 0){
			if (node2->tag == 1){
				temp_addr = node2->addr + node2->size;
				node1->addr = solve_conflict(temp_addr, graph, t[i], 1);
			} else {
				temp_addr = node2->addr - node1->size;
				node1->addr = solve_conflict(temp_addr, graph, t[i], -1);
			}
			node1->done = 1;
		}
	}

	return;
}
