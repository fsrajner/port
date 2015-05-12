using System;
using System.Collections.Generic;
using System.Data.Entity;
using System.Data.SqlClient;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Application4
{
    class Program
    {
        static void Main(string[] args)
        {
             DB db= new DB();
            Database.SetInitializer<DB>(null);
             db.Database.Log = (msg) => Console.WriteLine(msg);

             Product pr = new Product() { ProductName = "gránátalma" };
             Category c = new Category() { CategoryName = "háborús bűnös" };
             pr.Category = c;

            // db.Products.Add(pr);
             
             db.SaveChanges();

          //   db.Categories.ToList();

           // var product = db.Products.Single(p => p.ProductID == 1);

            var query = db.Products.Select(p => new {p.ProductName, p.Category.CategoryName, Count=p.Order_Details.Count()});   // direkt azt kérdezem le, ami kell nekem
                //from p in db.Products
                //where p.UnitPrice > 5
                //orderby -p.UnitPrice
                //select p;

            foreach (var item in query.Where(p => p.ProductName.StartsWith("g")))
            {
                Console.WriteLine(item.CategoryName+ "\t"+item.ProductName+ "\t"+item.Count);
            }
            Console.ReadKey();
        }
    }
}

/** Régi adatelérő logika
void sznikakoslekerdezes()
{ 
   string connString=  "Data Source=\"d3.aut.bme.hu, 8085\";"+
                    "Initial Catalog=Northwind;Persist Security Info=True;"+
                    "User ID=tanf;Password=TfUser14F2";


var conn = new SqlConnection(connString);
var cmd = new SqlCommand("Select * FROM products", conn);
conn.Open();
var reader = cmd.ExecuteReader();
while (reader.Read())
{
    Console.WriteLine(reader["ProductName"] + "\t" + reader["unitPrice"]);
}
 
}
*/

/** új, menő lekérdezés, erőse típusosan        
  void new()
        {
            DB db = new DB();
            Database.SetInitializer<DB>(null);
           // db.Database.Log = (msg) => Console.WriteLine(msg);

            var query =
                from p in db.Products
                    where p.UnitPrice>5
                    orderby -p.UnitPrice
                    select p;

            foreach (var item in query.Where(p=>p.ProductName.StartsWith("A")))
            {
                 Console.WriteLine(item.ProductName+"\t"+item.UnitPrice);
            }
            Console.ReadKey();
        }
  */