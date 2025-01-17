/**
 * Copyright (c) 2024 OceanBase
 * OceanBase CE is licensed under Mulan PubL v2.
 * You can use this software according to the terms and conditions of the Mulan PubL v2.
 * You may obtain a copy of Mulan PubL v2 at:
 *          http://license.coscl.org.cn/MulanPubL-2.0
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PubL v2 for more details.
 */

#ifndef _OB_TABLE_LS_EXECUTE_PROCESSOR_H
#define _OB_TABLE_LS_EXECUTE_PROCESSOR_H
#include "rpc/obrpc/ob_rpc_proxy.h"
#include "rpc/obrpc/ob_rpc_processor.h"
#include "share/table/ob_table_rpc_proxy.h"
#include "ob_table_rpc_processor.h"
#include "ob_table_context.h"
#include "ob_table_batch_service.h"


namespace oceanbase
{
namespace observer
{


struct ObTableHbaseMutationInfo : public ObTableInfoBase
{
  ObTableHbaseMutationInfo(common::ObIAllocator &allocator):
    ctx_(allocator) {}
  TO_STRING_KV(K(ctx_));
  table::ObTableCtx ctx_;
};


/// @see RPC_S(PR5 ls_op_execute, obrpc::OB_TABLE_API_LS_EXECUTE, (table::ObTableLSOpRequest), table::ObTableLSOpResult);
class ObTableLSExecuteP: public ObTableRpcProcessor<obrpc::ObTableRpcProxy::ObRpc<obrpc::OB_TABLE_API_LS_EXECUTE> >
{
  typedef ObTableRpcProcessor<obrpc::ObTableRpcProxy::ObRpc<obrpc::OB_TABLE_API_LS_EXECUTE> > ParentType;
public:
  explicit ObTableLSExecuteP(const ObGlobalContext &gctx);
  virtual ~ObTableLSExecuteP();
  virtual int deserialize() override;

protected:
  virtual int check_arg() override;
  int init_multi_schema_info(const ObString& arg_tablegroup_name);

  virtual int try_process() override;
  virtual void reset_ctx() override;
  virtual uint64_t get_request_checksum();
  virtual int before_process();
  virtual table::ObTableEntityType get_entity_type() override { return arg_.entity_type_; }
  virtual bool is_kv_processor() override { return true; }
private:
  int get_ls_id(ObLSID &ls_id, const share::schema::ObSimpleTableSchemaV2 *simple_table_schema);
  int check_arg_for_query_and_mutate(const table::ObTableSingleOp &single_op);
  int execute_tablet_op(const table::ObTableTabletOp &tablet_op,
                        uint64_t table_id,
                        table::ObKvSchemaCacheGuard *schema_cache_guard,
                        const ObSimpleTableSchemaV2 *simple_table_schema,
                        table::ObTableTabletOpResult &tablet_result);
  int execute_tablet_query_and_mutate(const table::ObTableTabletOp &tablet_op, table::ObTableTabletOpResult&tablet_result);
  int execute_tablet_query_and_mutate(const uint64_t table_id, const table::ObTableTabletOp &tablet_op, table::ObTableTabletOpResult&tablet_result);
  int execute_single_query_and_mutate(const uint64_t table_id,
                                      const common::ObTabletID tablet_id,
                                      const table::ObTableSingleOp &single_op,
                                      table::ObTableSingleOpResult &result);
  // batch service
  int execute_tablet_batch_ops(const table::ObTableTabletOp &tablet_op,
                                uint64_t table_id,
                                table::ObKvSchemaCacheGuard *schema_cache_guard,
                                const ObSimpleTableSchemaV2 *simple_table_schema,
                                table::ObTableTabletOpResult&tablet_result);

  int try_process_tablegroup(table::ObTableLSOpResult& ls_result);

  int modify_htable_quailfier_and_timestamp(const table::ObITableEntity *entity,
                                            const ObString& qualifier,
                                            int64_t timestamp,
                                            table::ObTableOperationType::Type type);


  int init_batch_ctx(table::ObTableBatchCtx &batch_ctx,
                    const table::ObTableTabletOp &tablet_op,
                    ObIArray<table::ObTableOperation> &table_operations,
                    uint64_t table_id,
                    table::ObKvSchemaCacheGuard *shcema_cache_guard,
                    const ObSimpleTableSchemaV2 *table_schema,
                    table::ObTableTabletOpResult &tablet_result);
  int init_tb_ctx(table::ObTableCtx &tb_ctx,
                  const table::ObTableTabletOp &tablet_op,
                  const table::ObTableOperation &table_operation,
                  table::ObKvSchemaCacheGuard *shcema_cache_guard,
                  const ObSimpleTableSchemaV2 *table_schema);

  int add_dict_and_bm_to_result_entity(const table::ObTableTabletOp &tablet_op,
                                       table::ObTableTabletOpResult &tablet_result);
  int execute_ls_op(table::ObTableLSOpResult &ls_result);

  int execute_ls_op_tablegroup(table::ObTableLSOpResult& ls_result);
  // void record_aft_rtn_rows(const table::ObTableLSOpResult &ls_result);

private:

  int find_and_add_op(ObIArray<std::pair<uint64_t, table::ObTableTabletOp>>& tablet_ops,
                        uint64_t real_table_id,
                        const table::ObTableSingleOp& single_op,
                        bool &is_found);
  int add_new_op(ObIArray<std::pair<uint64_t, table::ObTableTabletOp>>& tablet_ops,
                  uint64_t real_table_id, uint64_t real_tablet_id,
                  table::ObTableTabletOp& origin_tablet_op,
                  const table::ObTableSingleOp& single_op);
  int partition_single_op(table::ObTableTabletOp& tablet_op,
                          common::ObIArray<std::pair<uint64_t, table::ObTableTabletOp>>& tablet_ops);

  int construct_delete_family_op(const table::ObTableSingleOp &single_op,
                                  const ObTableHbaseMutationInfo &mutation_info,
                                  table::ObTableTabletOp &tablet_op);

private:
  common::ObArenaAllocator allocator_;
  ObArray<ObTableHbaseMutationInfo*> hbase_infos_;
  table::ObTableEntityFactory<table::ObTableSingleOpEntity> *default_entity_factory_;
  table::ObTableLSExecuteCreateCbFunctor cb_functor_;
  table::ObTableLSExecuteEndTransCb *cb_;
};

} // end namespace observer
} // end namespace oceanbase

#endif /* _OB_TABLE_BATCH_EXECUTE_PROCESSOR_H */
