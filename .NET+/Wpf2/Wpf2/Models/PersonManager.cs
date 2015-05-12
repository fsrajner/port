using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Wpf2.Models
{
    public class PersonManager
    {
        public static List<Person> Persons =
            Enumerable.Range(10, 30).Select(i => new Person { Name = "Jack", Age = i }).ToList();  //szépen legeneráljuk az emberkéket
    }
}
