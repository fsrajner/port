using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace Application5
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
        }

        private void bindingNavigatorMoveNextItem_Click(object sender, EventArgs e)
        {

        }

        private void Form1_Load(object sender, EventArgs e)
        {
            PersonList pl = new PersonList();
            for (int i = 0; i < 10; i++)
                pl.Add(new Person { Name = "Béla", Age = 20 + i });


            personListBindingSource.DataSource = pl;
        }

        private void personListBindingSource_CurrentChanged(object sender, EventArgs e)
        {

        }

        private void button1_Click(object sender, EventArgs e)
        {
            PersonList pl = (PersonList)personListBindingSource.DataSource;
            pl[0].Age++;
        }

    }
}
