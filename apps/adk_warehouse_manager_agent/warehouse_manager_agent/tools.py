from typing import List, Dict, Any
import psycopg2
from psycopg2.extras import RealDictCursor


def check_warehouse_availability(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Check availability of items across warehouses, including partial fulfillment options.

    Args:
        items: A list of items to check. Each item is a dictionary with keys: product_id, quantity.

    Returns:
        A dictionary containing:
        - can_fulfill_completely: bool indicating if all items can be fulfilled from at least one warehouse
        - warehouses_full_fulfillment: list of warehouses that can fulfill the entire order
        - warehouses_partial_fulfillment: list of warehouses with partial availability
        - unavailable_items: list of items that cannot be fulfilled from any warehouse
        - details: detailed breakdown per warehouse with availability for each item
    """
    import psycopg2
    from psycopg2.extras import RealDictCursor

    PG_CONN_INFO = {
        "dbname": "tools_database",
        "user": "langgraph_user",
        "password": "langgraph_password",
        "host": "localhost",
        "port": 5432,
    }

    try:
        conn = psycopg2.connect(**PG_CONN_INFO)
    except Exception as e:
        return {"error": f"Database connection failed: {e}"}

    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            all_product_ids = [item["product_id"] for item in items]
            cur.execute(
                """
                SELECT warehouse_id, product_id, available_quantity, warehouse_location, warehouse_name
                FROM warehouses.inventory
                WHERE product_id = ANY(%s)
                """,
                (all_product_ids,),
            )
            inventory_rows = cur.fetchall()
    except Exception as e:
        conn.close()
        return {"error": f"Inventory query failed: {e}"}

    warehouse_map: Dict[str, Any] = {}
    for row in inventory_rows:
        wid = row["warehouse_id"]
        pid = row["product_id"]
        avail = row["available_quantity"]
        loc = row.get("warehouse_location")
        name = row.get("warehouse_name")
        if wid not in warehouse_map:
            warehouse_map[wid] = {
                "items": {},
                "warehouse_location": loc,
                "warehouse_name": name,
            }
        warehouse_map[wid]["items"][pid] = avail

    warehouses_full_fulfillment = []
    warehouses_partial_fulfillment = []
    details: Dict[str, Any] = {}
    unavailable_items = []
    fully_unavailable_products: set = set()

    for wid, wdata in warehouse_map.items():
        can_fulfill_all = True
        partial_fulfillment = False
        warehouse_detail = {
            "warehouse_id": wid,
            "warehouse_location": wdata["warehouse_location"],
            "warehouse_name": wdata["warehouse_name"],
            "items": [],
        }

        for item in items:
            pid = item["product_id"]
            qty = item["quantity"]
            available = wdata["items"].get(pid, 0)

            if available >= qty:
                status = "available"
            elif available > 0:
                status = "partial"
                can_fulfill_all = False
                partial_fulfillment = True
            else:
                status = "unavailable"
                available = 0
                can_fulfill_all = False
                partial_fulfillment = True

            warehouse_detail["items"].append({
                "product_id": pid,
                "requested_quantity": qty,
                "available_quantity": available,
                "status": status,
            })

        details[wid] = warehouse_detail
        if can_fulfill_all:
            warehouses_full_fulfillment.append(wid)
        elif partial_fulfillment:
            warehouses_partial_fulfillment.append(wid)

    for item in items:
        pid = item["product_id"]
        qty = item["quantity"]
        found = any(
            wdata["items"].get(pid, 0) >= qty for wdata in warehouse_map.values()
        )
        if not found:
            max_anywhere = max(
                (wdata["items"].get(pid, 0) for wdata in warehouse_map.values()),
                default=0,
            )
            if max_anywhere == 0:
                fully_unavailable_products.add(pid)
            unavailable_items.append({
                "product_id": pid,
                "requested_quantity": qty,
                "max_available_quantity": max_anywhere,
                "status": "fully_unavailable" if max_anywhere == 0 else "partially_available",
            })

    conn.close()

    return {
        "can_fulfill_completely": len(warehouses_full_fulfillment) > 0,
        "warehouses_full_fulfillment": warehouses_full_fulfillment,
        "warehouses_partial_fulfillment": warehouses_partial_fulfillment,
        "unavailable_items": unavailable_items,
        "details": details,
    }

def reserve_warehouse_items(reservations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Reserve items from multiple warehouses. Successful reservations are committed
    even if some items fail (savepoint-per-item).

    Args:
        reservations: A list of reservations. Each reservation is a dictionary with keys:
                     - warehouse_id: The warehouse to reserve from
                     - product_id: The product to reserve
                     - quantity: The quantity to reserve

    Returns:
        A dictionary containing:
        - success: bool indicating if all reservations were successful
        - reserved_items: list of successfully reserved items
        - failed_items: list of items that could not be reserved
    """
    import psycopg2

    def get_db_conn():
        return psycopg2.connect(
            dbname="tools_database",
            user="langgraph_user",
            password="langgraph_password",
            host="localhost",
            port=5432,
        )

    reserved_items: list = []
    failed_items: list = []

    conn = None
    try:
        conn = get_db_conn()
        conn.autocommit = False
        cur = conn.cursor()

        for idx, reservation in enumerate(reservations):
            warehouse_id = reservation["warehouse_id"]
            product_id = reservation["product_id"]
            quantity = reservation["quantity"]

            savepoint = f"sp_{idx}"
            cur.execute(f"SAVEPOINT {savepoint}")

            try:
                cur.execute(
                    """
                    SELECT id, available_quantity, reserved_quantity, total_quantity
                    FROM warehouses.inventory
                    WHERE warehouse_id=%s AND product_id=%s
                    FOR UPDATE
                    """,
                    (warehouse_id, product_id),
                )
                row = cur.fetchone()

                if row is None:
                    cur.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
                    failed_items.append({
                        "warehouse_id": warehouse_id,
                        "product_id": product_id,
                        "requested_quantity": quantity,
                        "reason": "Item not found",
                    })
                    continue

                inv_id, available_quantity, reserved_quantity, total_quantity = row

                if available_quantity < quantity:
                    cur.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
                    failed_items.append({
                        "warehouse_id": warehouse_id,
                        "product_id": product_id,
                        "requested_quantity": quantity,
                        "available_quantity": available_quantity,
                        "reason": "Not enough available quantity",
                    })
                    continue

                cur.execute(
                    """
                    UPDATE warehouses.inventory
                    SET reserved_quantity = reserved_quantity + %s
                    WHERE id=%s
                    RETURNING reserved_quantity, available_quantity
                    """,
                    (quantity, inv_id),
                )
                updated = cur.fetchone()
                cur.execute(f"RELEASE SAVEPOINT {savepoint}")
                reserved_items.append({
                    "warehouse_id": warehouse_id,
                    "product_id": product_id,
                    "reserved_quantity": quantity,
                    "total_reserved": updated[0],
                    "remaining_available": updated[1],
                })

            except Exception as item_err:
                cur.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
                failed_items.append({
                    "warehouse_id": warehouse_id,
                    "product_id": product_id,
                    "requested_quantity": quantity,
                    "reason": str(item_err),
                })

        conn.commit()

    except Exception as e:
        if conn:
            conn.rollback()
        reserved_items = []
        failed_items.append({"error": str(e)})
    finally:
        if conn:
            conn.close()

    return {
        "success": len(failed_items) == 0,
        "reserved_items": reserved_items,
        "failed_items": failed_items,
    }